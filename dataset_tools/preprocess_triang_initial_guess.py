import os
import argparse
import numpy as np
import open3d as o3d
from collections import defaultdict
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="experiments/h36m-triang/point_cloud/iteration_0", help="Path to the metrabs prediction directory.")
parser.add_argument("--output_dir", type=str, default="../data/h36m", help="Path to save the reorganized dataset.")
args = parser.parse_args()

# Define new directory structures
input_dir = args.input_dir
output_root = args.output_dir
output_3d = os.path.join(output_root, "initial_guess/triang_gt")

os.makedirs(output_3d, exist_ok=True)

grouped_entries = defaultdict(list)

for entry in os.listdir(input_dir):
    print(f"Processing entry: {entry}")
    if entry.endswith('.ply'):
        parts = entry.split('_')
        if len(parts) >= 2:
            subject = parts[0]
            activity = parts[1] # + ('_' + parts[2] if len(parts) > 2 else '')
            grouped_entries[(subject, activity)].append(entry)

for (subject, activity), entries in grouped_entries.items():
    if "cpn" in input_dir:
        if subject == 'S11' and activity == 'Directions':
                    continue
        
    subject_dir = os.path.join(output_3d, subject)
    os.makedirs(subject_dir, exist_ok=True)
    
    activity_dir = os.path.join(subject_dir, activity)
    os.makedirs(activity_dir, exist_ok=True)

    data = []
    
    for entry in sorted(entries):
        input_path = os.path.join(input_dir, entry)
        output_path = os.path.join(activity_dir, entry)

        pcd = o3d.io.read_point_cloud(input_path)
        vertices = np.asarray(pcd.points)
        data.append(vertices)
    
    data_arr = np.array(data)
    npz_path = os.path.join(activity_dir, "poses.npz")
    np.savez(npz_path, poses3d=data_arr)

print(f"Done, data saved to {output_root}")


        
