#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import os
import logging
from argparse import ArgumentParser
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
import pycolmap
import pinocchio as pin

# Install: uv pip install pin pycolmap scipy numpy

parser = ArgumentParser("Bimanual robot Gaussian Splatting converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--joint_angles", type=str, required=True, 
                    help="Path to joint angles (Nx(DOF) numpy array for full bimanual robot)")
parser.add_argument("--urdf", type=str, required=True, help="URDF file for bimanual robot")
parser.add_argument("--end_effector_frame_arm1", type=str, required=True,
                    help="Name of end effector frame for arm 1 in URDF")
parser.add_argument("--end_effector_frame_arm2", type=str, required=True,
                    help="Name of end effector frame for arm 2 in URDF")
parser.add_argument("--hand_eye_transform_arm1", type=str, required=True,
                    help="4x4 transform from end effector to camera for arm 1 (.npy file)")
parser.add_argument("--hand_eye_transform_arm2", type=str, required=True,
                    help="4x4 transform from end effector to camera for arm 2 (.npy file)")
# Camera intrinsics for both cameras
parser.add_argument("--width_cam1", type=int, required=True)
parser.add_argument("--height_cam1", type=int, required=True)
parser.add_argument("--fx_cam1", type=float, required=True)
parser.add_argument("--fy_cam1", type=float, required=True)
parser.add_argument("--cx_cam1", type=float, required=True)
parser.add_argument("--cy_cam1", type=float, required=True)
parser.add_argument("--width_cam2", type=int, required=True)
parser.add_argument("--height_cam2", type=int, required=True)
parser.add_argument("--fx_cam2", type=float, required=True)
parser.add_argument("--fy_cam2", type=float, required=True)
parser.add_argument("--cx_cam2", type=float, required=True)
parser.add_argument("--cy_cam2", type=float, required=True)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
device = pycolmap.Device.auto if not args.no_gpu else pycolmap.Device.cpu


def compute_forward_kinematics_pinocchio(model, data, joint_angles, frame_name):
    """
    Compute forward kinematics using Pinocchio for a specific frame.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        joint_angles: array of joint angles (radians) for full robot
        frame_name: name of the end effector frame
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    # Update all frame placements
    pin.forwardKinematics(model, data, joint_angles)
    pin.updateFramePlacements(model, data)
    
    # Get frame by name
    frame_id = model.getFrameId(frame_name)
    
    # Get transform from world to frame
    T = data.oMf[frame_id]
    
    # Convert SE3 to numpy 4x4 matrix
    T_matrix = T.homogeneous
    
    return T_matrix


def robot_pose_to_colmap(T_robot):
    """
    Convert robot arm pose to COLMAP camera pose.
    
    Robot frame: X forward, Y left, Z up
    COLMAP frame: X right, Y down, Z forward
    
    Args:
        T_robot: 4x4 transformation matrix from world to camera
    
    Returns:
        qvec: quaternion [qw, qx, qy, qz] 
        tvec: translation [tx, ty, tz]
    """
    R_robot_to_colmap = np.array([
        [ 0, -1,  0],  # COLMAP X (right) = -Robot Y (left)
        [ 0,  0, -1],  # COLMAP Y (down) = -Robot Z (up)
        [ 1,  0,  0]   # COLMAP Z (forward) = Robot X (forward)
    ])
    
    R_robot = T_robot[:3, :3]
    t_robot = T_robot[:3, 3]
    
    # Convert rotation
    R_colmap = R_robot_to_colmap @ R_robot @ R_robot_to_colmap.T
    
    # Convert to quaternion (w, x, y, z)
    rot = Rotation.from_matrix(R_colmap)
    quat_xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
    qvec = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    # COLMAP uses projection center: tvec = -R^T * t
    t_colmap = R_robot_to_colmap @ t_robot
    tvec = -R_colmap.T @ t_colmap
    
    return qvec, tvec


def create_colmap_reconstruction_dual_arm(poses_cam1, poses_cam2, image_files_cam1, image_files_cam2,
                                          camera_params_cam1, camera_params_cam2, output_path):
    """
    Create COLMAP reconstruction from dual-arm camera poses.
    
    Args:
        poses_cam1: Nx4x4 array of camera poses for arm 1
        poses_cam2: Nx4x4 array of camera poses for arm 2
        image_files_cam1: list of image filenames for camera 1
        image_files_cam2: list of image filenames for camera 2
        camera_params_cam1/2: dict with camera intrinsics
        output_path: path to save reconstruction
    """
    os.makedirs(output_path, exist_ok=True)
    
    reconstruction = pycolmap.Reconstruction()
    
    # Add camera 1
    camera1 = pycolmap.Camera(
        model='PINHOLE',
        width=camera_params_cam1['width'],
        height=camera_params_cam1['height'],
        params=[camera_params_cam1['fx'], camera_params_cam1['fy'], 
                camera_params_cam1['cx'], camera_params_cam1['cy']]
    )
    camera_id1 = reconstruction.add_camera(camera1)
    
    # Add camera 2
    camera2 = pycolmap.Camera(
        model='PINHOLE',
        width=camera_params_cam2['width'],
        height=camera_params_cam2['height'],
        params=[camera_params_cam2['fx'], camera_params_cam2['fy'], 
                camera_params_cam2['cx'], camera_params_cam2['cy']]
    )
    camera_id2 = reconstruction.add_camera(camera2)
    
    image_id = 1
    
    # Add images from camera 1
    for T_cam, img_file in zip(poses_cam1, image_files_cam1):
        qvec, tvec = robot_pose_to_colmap(T_cam)
        
        image = pycolmap.Image(
            id=image_id,
            name=img_file,
            camera_id=camera_id1,
            cam_from_world=pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d(qvec),
                translation=tvec
            )
        )
        reconstruction.add_image(image)
        image_id += 1
    
    # Add images from camera 2
    for T_cam, img_file in zip(poses_cam2, image_files_cam2):
        qvec, tvec = robot_pose_to_colmap(T_cam)
        
        image = pycolmap.Image(
            id=image_id,
            name=img_file,
            camera_id=camera_id2,
            cam_from_world=pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d(qvec),
                translation=tvec
            )
        )
        reconstruction.add_image(image)
        image_id += 1
    
    reconstruction.write(output_path)
    logging.info(f"Created reconstruction with {reconstruction.num_images()} images from 2 cameras")
    
    return reconstruction


# ============================================================================
# Main processing
# ============================================================================

logging.info("Loading joint angles and hand-eye calibrations")

# Load joint angles for full bimanual robot
joint_angles = np.load(args.joint_angles)  # Shape: (N, total_dof)

# Load hand-eye transforms (end effector to camera)
T_ee_to_cam1 = np.load(args.hand_eye_transform_arm1)  # 4x4
T_ee_to_cam2 = np.load(args.hand_eye_transform_arm2)  # 4x4

# Load bimanual robot model using Pinocchio
logging.info(f"Loading bimanual robot model from URDF: {args.urdf}")
model = pin.buildModelFromUrdf(args.urdf)
data = model.createData()
logging.info(f"Bimanual robot: {model.nq} DOF, {len(model.frames)} frames")

# List available frames for reference
logging.info("Available frames in URDF:")
for i, frame in enumerate(model.frames):
    logging.info(f"  [{i}] {frame.name}")

# Verify the end effector frames exist
try:
    frame_id1 = model.getFrameId(args.end_effector_frame_arm1)
    frame_id2 = model.getFrameId(args.end_effector_frame_arm2)
    logging.info(f"End effector frames found: {args.end_effector_frame_arm1} (id={frame_id1}), "
                 f"{args.end_effector_frame_arm2} (id={frame_id2})")
except Exception as e:
    logging.error(f"Could not find end effector frames: {e}")
    logging.error("Please check the frame names in your URDF")
    exit(1)

# Verify joint angles shape matches model DOF
if joint_angles.shape[1] != model.nq:
    logging.error(f"Joint angles dimension ({joint_angles.shape[1]}) doesn't match "
                  f"robot DOF ({model.nq})")
    exit(1)

# Compute forward kinematics for both end effectors
logging.info("Computing forward kinematics for both arms")
poses_ee1 = []
poses_ee2 = []

for q in joint_angles:
    # Compute FK for arm 1 end effector
    T_ee1 = compute_forward_kinematics_pinocchio(model, data, q, args.end_effector_frame_arm1)
    poses_ee1.append(T_ee1)
    
    # Compute FK for arm 2 end effector (same joint config, different frame)
    T_ee2 = compute_forward_kinematics_pinocchio(model, data, q, args.end_effector_frame_arm2)
    poses_ee2.append(T_ee2)

poses_ee1 = np.array(poses_ee1)
poses_ee2 = np.array(poses_ee2)

# Apply hand-eye calibration to get camera poses
poses_cam1 = np.array([T_ee @ T_ee_to_cam1 for T_ee in poses_ee1])
poses_cam2 = np.array([T_ee @ T_ee_to_cam2 for T_ee in poses_ee2])

logging.info(f"Computed {len(poses_cam1)} poses for camera 1 and {len(poses_cam2)} poses for camera 2")

# Get image files
image_path_cam1 = os.path.join(args.source_path, "input", "cam1")
image_path_cam2 = os.path.join(args.source_path, "input", "cam2")

image_files_cam1 = sorted([f"cam1/{f}" for f in os.listdir(image_path_cam1) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_files_cam2 = sorted([f"cam2/{f}" for f in os.listdir(image_path_cam2)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if len(poses_cam1) != len(image_files_cam1):
    logging.error(f"Mismatch: {len(poses_cam1)} poses vs {len(image_files_cam1)} images for cam1")
    exit(1)

if len(poses_cam2) != len(image_files_cam2):
    logging.error(f"Mismatch: {len(poses_cam2)} poses vs {len(image_files_cam2)} images for cam2")
    exit(1)

# Setup paths
os.makedirs(args.source_path + "/distorted/sparse/0", exist_ok=True)
database_path = args.source_path + "/distorted/database.db"
image_path = args.source_path + "/input"

# Camera parameters
camera_params_cam1 = {
    'width': args.width_cam1, 'height': args.height_cam1,
    'fx': args.fx_cam1, 'fy': args.fy_cam1,
    'cx': args.cx_cam1, 'cy': args.cy_cam1
}

camera_params_cam2 = {
    'width': args.width_cam2, 'height': args.height_cam2,
    'fx': args.fx_cam2, 'fy': args.fy_cam2,
    'cx': args.cx_cam2, 'cy': args.cy_cam2
}

# Create reconstruction from known poses
sparse_path = args.source_path + "/distorted/sparse/0"
reconstruction = create_colmap_reconstruction_dual_arm(
    poses_cam1, poses_cam2, 
    image_files_cam1, image_files_cam2,
    camera_params_cam1, camera_params_cam2,
    sparse_path
)

# Feature extraction
try:
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_path,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,  # Different cameras per folder
        sift_options=pycolmap.SiftExtractionOptions(use_gpu=not args.no_gpu),
        device=device
    )
    logging.info("Feature extraction completed")
except Exception as e:
    logging.error(f"Feature extraction failed: {e}")
    exit(1)

# Feature matching
try:
    pycolmap.match_exhaustive(
        database_path=database_path,
        sift_options=pycolmap.SiftMatchingOptions(use_gpu=not args.no_gpu),
        device=device
    )
    logging.info("Feature matching completed")
except Exception as e:
    logging.error(f"Feature matching failed: {e}")
    exit(1)

# Triangulate points from known poses
try:
    pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=database_path,
        image_path=image_path,
        output_path=sparse_path
    )
    reconstruction.write(sparse_path)
    logging.info(f"Triangulation completed: {reconstruction.num_points3D()} 3D points")
except Exception as e:
    logging.error(f"Triangulation failed: {e}")
    exit(1)

# Image undistortion
try:
    pycolmap.undistort_images(
        output_path=args.source_path,
        input_path=sparse_path,
        image_path=image_path,
        output_type="COLMAP"
    )
    logging.info("Image undistortion completed")
except Exception as e:
    logging.error(f"Image undistortion failed: {e}")
    exit(1)

# Move files to sparse/0
files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)

for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    if os.path.exists(source_file):
        shutil.move(source_file, destination_file)

# Resize images if requested
if args.resize:
    logging.info("Resizing images...")
    for scale, folder in [(50, "images_2"), (25, "images_4"), (12.5, "images_8")]:
        os.makedirs(args.source_path + f"/{folder}", exist_ok=True)
        
        files = os.listdir(args.source_path + "/images")
        for file in files:
            source_file = os.path.join(args.source_path, "images", file)
            destination_file = os.path.join(args.source_path, folder, file)
            shutil.copy2(source_file, destination_file)
            
            exit_code = os.system(f"{magick_command} mogrify -resize {scale}% {destination_file}")
            if exit_code != 0:
                logging.error(f"{scale}% resize failed")
                exit(exit_code)

print("Done! Ready for Gaussian splatting training.")
