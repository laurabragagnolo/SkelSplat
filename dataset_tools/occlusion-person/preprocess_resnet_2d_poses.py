import os
import argparse
import numpy as np
import open3d as o3d
from collections import defaultdict
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="occlusion_person_2d_preds.npz", help="Path to the prediction directory.")
parser.add_argument("--output_dir", type=str, default="../../data/occlusion-person", help="Path to save the reorganized dataset.")
args = parser.parse_args()

# Define new directory structures
input_file = args.input_file
output_root = args.output_dir
output_2d = os.path.join(output_root, "2d_resnet")

os.makedirs(output_2d, exist_ok=True)

data = np.load(input_file, allow_pickle=True)
if 'preds' in data:
    preds = data['preds']
else:
    raise ValueError("Input file does not contain 'preds' key.")

print(f"Loaded {preds.shape} predictions from {input_file}")

subject_path = os.path.join(output_2d, "S0", "validation")
os.makedirs(subject_path, exist_ok=True)

print(f"{preds.shape} predictions")
for cam_id in range(8):
    cam_path = os.path.join(subject_path, str(cam_id))
    os.makedirs(cam_path, exist_ok=True)

    poses2d = preds[cam_id::8, :, :2]
    np.savez(os.path.join(cam_path, "poses.npz"), poses2d=poses2d)

    print(f"Saved 2D poses for camera {cam_id} with shape {poses2d.shape} in {cam_path}")

print(f"Saved 2D poses")


