import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="data_2d_h36m_cpn_ft_h36m_dbb/positions_2d.npy", help="Path to CPN predictions.")
parser.add_argument("--output_dir", type=str, default="../../data/h36m", help="Path to save the reorganized dataset.")
args = parser.parse_args()

input_file = args.input_file
output_root = args.output_dir
output_2d = os.path.join(output_root, "2d_cpn")

os.makedirs(output_2d, exist_ok=True)

data_cpn = np.load(input_file, allow_pickle=True).item()
metadata = np.load("data_2d_h36m_cpn_ft_h36m_dbb/metadata.npy", allow_pickle=True)
print(metadata)

# Iterate through subjects (S9, S11, etc.)
for subject in ["S9", "S11"]:
    subject_path = os.path.join(output_2d, subject)
    os.makedirs(subject_path, exist_ok=True)

    # Process each activity (e.g., Directions)
    for activity in sorted(data_cpn[subject].keys()):
        print(activity)
        activity_path = os.path.join(subject_path, activity)
        os.makedirs(activity_path, exist_ok=True)
        poses_2d = data_cpn[subject][activity]

        for i, cam_name in enumerate(["54138969", "55011271", "58860488", "60457274"]):
            output2d_path = os.path.join(output_2d, subject, activity, cam_name)
            os.makedirs(output2d_path, exist_ok=True)

            poses_cam = poses_2d[i]
            poses_cam = np.array(poses_cam).reshape(-1, 17, 2)
            poses_cam_step = np.array([poses_cam[j] for j in range(0, len(poses_cam), 64)])

            np.savez(os.path.join(output2d_path, "poses.npz"), poses2d=poses_cam_step)