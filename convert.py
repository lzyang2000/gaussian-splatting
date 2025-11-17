#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
import pycolmap

parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()

magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"

# Set device for GPU usage
device = pycolmap.Device.auto if not args.no_gpu else pycolmap.Device.cpu

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)
    
    database_path = args.source_path + "/distorted/database.db"
    image_path = args.source_path + "/input"
    
    ## Feature extraction
    try:
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_path,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model=args.camera,
            sift_options=pycolmap.SiftExtractionOptions(use_gpu=not args.no_gpu),
            device=device
        )
        logging.info("Feature extraction completed successfully.")
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        exit(1)
    
    ## Feature matching
    try:
        pycolmap.match_exhaustive(
            database_path=database_path,
            sift_options=pycolmap.SiftMatchingOptions(use_gpu=not args.no_gpu),
            device=device
        )
        logging.info("Feature matching completed successfully.")
    except Exception as e:
        logging.error(f"Feature matching failed: {e}")
        exit(1)
    
    ### Bundle adjustment / Mapping
    output_path = args.source_path + "/distorted/sparse"
    
    try:
        # Configure mapper options with lower tolerance for faster BA
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.ba_global_function_tolerance = 0.000001
        
        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_path,
            output_path=output_path,
            options=mapper_options
        )
        
        if len(maps) == 0:
            logging.error("Mapper failed to generate any reconstructions.")
            exit(1)
            
        # Write the largest reconstruction
        maps[0].write(output_path)
        logging.info(f"Mapper completed successfully with {len(maps)} reconstruction(s).")
    except Exception as e:
        logging.error(f"Mapper failed: {e}")
        exit(1)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
try:
    pycolmap.undistort_images(
        output_path=args.source_path,
        input_path=args.source_path + "/distorted/sparse/0",
        image_path=args.source_path + "/input",
        output_type="COLMAP"
    )
    logging.info("Image undistortion completed successfully.")
except Exception as e:
    logging.error(f"Image undistortion failed: {e}")
    exit(1)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)

# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if args.resize:
    print("Copying and resizing...")
    
    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)
        
        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
