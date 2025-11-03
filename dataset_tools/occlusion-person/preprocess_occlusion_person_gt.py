import os
import numpy as np
import argparse
import pickle as pkl
import json


def convert_numpy_to_list(obj):
    
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    

# Argument parser
parser = argparse.ArgumentParser(description="Reorganize Occlusion-Person dataset.")
parser.add_argument("--pkl_file", type=str, help="Path to the pickle file with annotations.")
parser.add_argument("--output_dir", type=str, default="../../data/occlusion-person", help="Path to save the reorganized dataset.")
args = parser.parse_args()

# Define new directory structures
output_root = args.output_dir
output_3d = os.path.join(output_root, "3d_gt")
output_2d = os.path.join(output_root, "2d_gt")

os.makedirs(output_3d, exist_ok=True)
os.makedirs(output_2d, exist_ok=True)

# Load the pickle file
with open(args.pkl_file, 'rb') as f:
    data = pkl.load(f)

joints_2d_gt = []
joints_3d_gt = []
camera_list = []

poses_2d_gt = {}
poses_3d_gt = {}
cameras = {}
cameras_to_save = {}

for d in data:
    joints_2d_gt.append(d["joints_2d"])
    joints_3d_gt.append(d["joints_gt"])
    camera_list.append(d["camera"])

joints_2d_gt = np.array(joints_2d_gt)
joints_3d_gt = np.array(joints_3d_gt)

print(f"Loaded {joints_2d_gt.shape} 2D joints and {joints_3d_gt.shape} 3D joints.")

for camera_id in range(8):
    output_2d = os.path.join(output_root, "2d_gt", "S0", str(camera_id))
    os.makedirs(output_2d, exist_ok=True)

    poses_2d_gt[camera_id] = joints_2d_gt[camera_id::8, :, :2]
    poses_2d_gt[camera_id] = np.array(poses_2d_gt[camera_id])[::5, ...]  # Downsample as in AdaFuse

    np.savez(os.path.join(output_2d, "poses.npz"), poses2d=poses_2d_gt[camera_id])

    print(f"Saved 2D poses for camera {camera_id} with shape {poses_2d_gt[camera_id].shape} in {output_2d}")


output_3d = os.path.join(output_root, "3d_gt", "S0", "validation")
os.makedirs(output_3d, exist_ok=True)

poses_3d_gt[camera_id] = joints_3d_gt[0::8, :, :3]
poses_3d_gt[camera_id] = np.array(poses_3d_gt[camera_id])[::5, ...]   # Downsample as in AdaFuse

np.savez(os.path.join(output_3d, "poses.npz"), poses3d=poses_3d_gt[camera_id])

print(f"Saved 3D poses with shape {poses_3d_gt[camera_id].shape} in {output_3d}")


# Reorganize cameras
for camera_id in range(8):
    cameras[camera_id] = camera_list[camera_id::8][::5]
    print(f"Camera {camera_id} has {len(cameras[camera_id])} frames.")

for f in range(len(cameras[0])):

    cameras_to_save[f] = []
    for camera_id in range(8):
        cameras_to_save[f].append(convert_numpy_to_list(cameras[camera_id][f])) 
    print(f"Frame {f} has {len(cameras_to_save[f])} frames.")

# Save cameras to json
cameras_file = os.path.join(output_root, "cameras.json")
with open(cameras_file, 'w') as f:
    json.dump(cameras_to_save, f, indent=4)