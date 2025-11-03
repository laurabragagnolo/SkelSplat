import os
import numpy as np
import cdflib
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Preprocess h36m GT dataset.")
parser.add_argument("--root_dir", type=str, default="path/to/your/h36m", help="Path to the h36m dataset directory.")
parser.add_argument("--output_dir", type=str, default="../../data/h36m", help="Path to save the reorganized dataset.")
args = parser.parse_args()

# Define new directory structures
root_dir = args.root_dir
output_root = args.output_dir
output_3d = os.path.join(output_root, "3d_gt")
output_2d = os.path.join(output_root, "2d_gt")

os.makedirs(output_3d, exist_ok=True)
os.makedirs(output_2d, exist_ok=True)

i_relevant_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

def process_cdf_to_npz(cdf_path, save_path):
    """Reads a CDF file and saves the extracted data as an NPZ file."""
    cdf_data = cdflib.CDF(cdf_path)
    keys = cdf_data.cdf_info().zVariables  # Get variable names
    
    if not keys:
        print(f"Warning: No variables found in {cdf_path}")
        return
    
    pose_data = cdf_data.varget(keys[0])  
    
    if "3d" in save_path:
        pose_data = pose_data.reshape(-1, 32, 3)
    else:
        pose_data = pose_data.reshape(-1, 32, 2)
    
    pose_data = pose_data[:, i_relevant_joints, :]
    print(f"Loaded {cdf_path} with shape {pose_data.shape}")
    # Save as NPZ
    np.savez_compressed(save_path, poses=pose_data)
    print(f"Saved {save_path}")

def process_npy_to_npz(npy_path, save_path):
    """Reads an .npy file and saves it as an NPZ file."""
    data = np.load(npy_path)
    print(npy_path)
    print(data[:10])
    np.savez_compressed(save_path, boxes=data)
    print(f"Saved {save_path}")

# Traverse subjects
for subject in sorted(os.listdir(root_dir)):
    subject_path = os.path.join(root_dir, subject)
    if not os.path.isdir(subject_path) or not subject.startswith("S"):
        continue  # Skip non-subject folders

    # Process 3D poses (D3_Positions)
    d3_path = os.path.join(subject_path, "MyPoseFeatures", "D3_Positions")
    if os.path.exists(d3_path):
        for cdf_file in os.listdir(d3_path):
            if not cdf_file.endswith(".cdf"):
                continue  # Skip non-CDF files
            
            # Extract action name (e.g., "Directions" from "Directions.cdf")
            action = os.path.splitext(cdf_file)[0]

            # Define new directory and filename
            new_subject_path = os.path.join(output_3d, subject, action)
            os.makedirs(new_subject_path, exist_ok=True)
            save_file = os.path.join(new_subject_path, "poses.npz")

            # Convert CDF to NPZ
            process_cdf_to_npz(os.path.join(d3_path, cdf_file), save_file)

    # Process 2D poses (D2_Positions)
    d2_path = os.path.join(subject_path, "MyPoseFeatures", "D2_Positions")
    if os.path.exists(d2_path):
        for cdf_file in os.listdir(d2_path):
            if not cdf_file.endswith(".cdf"):
                continue  # Skip non-CDF files
            
            # Extract action and camera code (e.g., "Directions.54138969.cdf" → "Directions", "54138969")
            filename_parts = cdf_file.split('.')
            if len(filename_parts) < 3:
                print(f"Skipping malformed filename: {cdf_file}")
                continue  # Skip files that don't match expected pattern
            
            action = filename_parts[0]  # "Directions"
            camera_code = filename_parts[1]  # "54138969"

            # Define new directory and filename
            new_subject_path = os.path.join(output_2d, subject, action, camera_code)
            os.makedirs(new_subject_path, exist_ok=True)
            save_file = os.path.join(new_subject_path, "poses.npz")

            # Convert CDF to NPZ
            process_cdf_to_npz(os.path.join(d2_path, cdf_file), save_file)

    # Process bounding boxes (BBoxes)
    bboxes_path = os.path.join(subject_path, "BBoxes")
    if os.path.exists(bboxes_path):
        for npy_file in os.listdir(bboxes_path):
            if not npy_file.endswith(".npy"):
                continue  # Skip non-NPY files
            
            # Extract action and camera code (e.g., "Directions.54138969.npy" → "Directions", "54138969")
            filename_parts = npy_file.split('.')
            if len(filename_parts) < 3:
                print(f"Skipping malformed filename: {npy_file}")
                continue  # Skip files that don't match expected pattern
            
            action = filename_parts[0]  # "Directions"
            camera_code = filename_parts[1]  # "54138969"

            # Define corresponding 2D pose folder
            new_subject_path = os.path.join(output_2d, subject, action, camera_code)
            os.makedirs(new_subject_path, exist_ok=True)
            save_file = os.path.join(new_subject_path, "boxes.npz")

            # Convert NPY to NPZ
            process_npy_to_npz(os.path.join(bboxes_path, npy_file), save_file)
