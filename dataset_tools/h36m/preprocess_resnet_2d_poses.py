import os
import argparse
import numpy as np
import open3d as o3d
from collections import defaultdict
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="2d_preds.npz", help="Path to the metrabs prediction directory.")
parser.add_argument("--output_dir", type=str, default="../../data/h36m", help="Path to save the reorganized dataset.")
args = parser.parse_args()

# Define new directory structures
input_file = args.input_file
output_root = args.output_dir
output_2d = os.path.join(output_root, "2d_resnet")

os.makedirs(output_2d, exist_ok=True)

activities_S9 = [
    "Directions 1", "Directions", "Discussion 1", "Discussion 2", "Eating 1", "Eating", "Greeting 1", "Greeting", "Phoning 1", "Phoning", "Posing 1", "Posing", "Purchases 1", "Purchases", "Sitting 1", "Sitting", "SittingDown", "SittingDown 1",
    "Smoking 1", "Smoking", "Photo 1", "Photo", "Waiting 1", "Waiting", "Walking 1", "Walking", "WalkDog 1", "WalkDog", "WalkTogether 1", "WalkTogether"
]

activities_S11 = [
    "Directions 1", "Directions", "Discussion 1", "Discussion 2", "Eating 1", "Eating", "Greeting 2", "Greeting", "Phoning 3", "Phoning 2", "Posing 1", "Posing", "Purchases 1", "Purchases", "Sitting 1", "Sitting", "SittingDown", "SittingDown 1",
    "Smoking 2", "Smoking", "Photo 1", "Photo", "Waiting 1", "Waiting", "Walking 1", "Walking", "WalkDog 1", "WalkDog", "WalkTogether 1", "WalkTogether"
]

# To divide the predictions correctly
activities_length = [37, 43, 92, 83, 42, 42, 43, 23, 60, 52, 31, 31, 20, 24, 48, 47, 46, 25, 69, 68, 23, 37, 26, 52, 39, 26, 35, 35, 27, 27, 25, 29, 42, 35,
                      36, 35, 27, 29, 53, 55, 24, 22, 17, 17, 30, 35, 29, 32, 44, 38, 25, 32, 36, 36, 26, 26, 19, 23, 29, 22]


data = np.load(input_file, allow_pickle=True)
if 'preds' in data:
    preds = data['preds']
else:
    raise ValueError("Input file does not contain 'preds' key.")

print(f"Loaded {preds.shape} predictions from {input_file}")

cnt = 0
cnt_activity = 0

for subject in ("S9", "S11"):
    subject_path = os.path.join(output_2d, subject)
    os.makedirs(subject_path, exist_ok=True)

    if subject == "S9":
        activities = activities_S9
    elif subject == "S11":
        activities = activities_S11

    for activity in activities:
        activity_path = os.path.join(subject_path, activity)
        os.makedirs(activity_path, exist_ok=True)

        preds_activity = preds[cnt:cnt + activities_length[cnt_activity] * 4]
        cnt += activities_length[cnt_activity] * 4
        cnt_activity += 1
        print(f"Processing {subject} - {activity} with {len(preds_activity)} predictions")
        for i, cam_name in enumerate(["54138969", "55011271", "58860488", "60457274"]):
            cam_path = os.path.join(activity_path, cam_name)
            os.makedirs(cam_path, exist_ok=True)

            poses2d = preds_activity[i::4, :, :2]

            np.savez(os.path.join(cam_path, "poses.npz"), poses2d=poses2d)


