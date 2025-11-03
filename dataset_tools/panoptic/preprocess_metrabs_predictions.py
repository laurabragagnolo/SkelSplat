import os
import argparse
import numpy as np
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="path/to/your/panoptic_preds", help="Path to predictions.")
parser.add_argument("--output_dir", type=str, default="../../data/panoptic", help="Path to save the reorganized dataset.")
parser.add_argument("--path_3d", type=str, default="", help="Name for dir for 3D data.")
parser.add_argument("--path_2d", type=str, default="", help="Name for dir for 2D data.")
args = parser.parse_args()

# Define new directory structures
input_file = args.input_dir
output_root = args.output_dir
path_3d = args.path_3d
path_2d = args.path_2d

output_3d = os.path.join(output_root, path_3d)
output_2d =  os.path.join(output_root, path_2d)

os.makedirs(output_3d, exist_ok=True)
os.makedirs(output_2d, exist_ok=True)

activities = ["171204_pose5", "171204_pose6"]

# Iterate through subjects (S9, S11, etc.)
for activity in activities:
    activity_path_3d = os.path.join(output_3d, "S0", activity)
    activity_path_2d = os.path.join(output_2d, "S0", activity)
    os.makedirs(activity_path_3d, exist_ok=True)
    os.makedirs(activity_path_2d, exist_ok=True)

    input_path = os.path.join(input_file, activity)
    for camera in os.listdir(input_path):
        camera_path_3d = os.path.join(activity_path_3d, camera)
        camera_path_2d = os.path.join(activity_path_2d, camera)
        os.makedirs(camera_path_3d, exist_ok=True)
        os.makedirs(camera_path_2d, exist_ok=True)
        
        source_path_3d = os.path.join(input_path, camera, "poses3d_world.npz")
        destination_path_3d = os.path.join(camera_path_3d, "poses.npz")
        shutil.copy2(source_path_3d, destination_path_3d)

        source_path_2d = os.path.join(input_path, camera, "poses2d.npz")
        destination_path_2d = os.path.join(camera_path_2d, "poses.npz")
        shutil.copy2(source_path_2d, destination_path_2d)

print(f"Processed activities: {activities}")
print(f"3D predictions saved to: {output_3d}")
print(f"2D predictions saved to: {output_2d}")



