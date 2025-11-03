import os
import argparse
import numpy as np


CAMERA_NAMES = ["54138969", "55011271", "58860488", "60457274"]


def preprocess_2d(input_dir: str, output_root: str):
    """
    Reads per-activity 2D files from `input_dir` and writes them into:
      {output_root}/2d_metrabs/{Sxx}/{Activity}/{CamID}/
        - poses.npz (poses2d)
    """
    output_2d = os.path.join(output_root, "2d_metrabs")
    os.makedirs(output_2d, exist_ok=True)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"--input_dir not found or not a directory: {input_dir}")

    for subject in sorted(os.listdir(input_dir)):
        subject_path = os.path.join(input_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        for activity in sorted(os.listdir(subject_path)):
            activity_path = os.path.join(subject_path, activity)
            if not os.path.isdir(activity_path):
                continue

            # Load arrays (expects the given keys/files)
            poses2d = np.load(os.path.join(activity_path, "poses2d.npz"))["poses2d"]

            # Create output
            for i, cam_name in enumerate(CAMERA_NAMES):
                out_cam_dir = os.path.join(output_2d, subject, activity, cam_name)
                os.makedirs(out_cam_dir, exist_ok=True)

                np.savez(os.path.join(out_cam_dir, "poses.npz"), poses2d=poses2d[i, :, :, :])

            print(f"Wrote: {subject}/{activity} -> {os.path.join(output_2d, subject, activity)}")

    print("2D Done.")


def preprocess_3d(preds_3d_file: str, output_root: str):
    """
    Splits a single 3D predictions .npz into per-activity/camera chunks and writes to:
      {output_root}/3d_metrabs_mono/{Sxx}/{Activity}/{CamID}/poses.npz (poses3d)
    Expects key 'coords3d_pred_world' in the npz.
    """
    output_3d = os.path.join(output_root, "3d_metrabs_mono")
    os.makedirs(output_3d, exist_ok=True)

    # 3D predictions need to be split per activity

    activities_S9 = [
        "Directions", "Directions 1","Discussion 1", "Discussion 2", "Eating", "Eating 1", "Greeting", "Greeting 1",
        "Phoning", "Phoning 1", "Photo", "Photo 1", "Posing", "Posing 1", "Purchases", "Purchases 1",
        "Sitting", "Sitting 1", "SittingDown", "SittingDown 1", "Smoking", "Smoking 1", "Waiting", "Waiting 1",
        "WalkDog", "WalkDog 1", "WalkTogether", "WalkTogether 1", "Walking", "Walking 1"
    ]

    activities_S11 = [
        "Directions", "Directions 1", "Discussion 1", "Discussion 2", "Eating", "Eating 1", "Greeting", "Greeting 2",
        "Phoning 2", "Phoning 3", "Photo", "Photo 1", "Posing", "Posing 1", "Purchases", "Purchases 1", "Sitting",
        "Sitting 1", "SittingDown", "SittingDown 1", "Smoking", "Smoking 2", "Waiting", "Waiting 1", "WalkDog",
        "WalkDog 1", "WalkTogether", "WalkTogether 1", "Walking", "Walking 1"
    ]

    activities_length = [
        43, 37, 92, 83, 42, 42, 23, 43, 52, 60, 37, 23, 31, 31, 24, 20, 47, 48, 46, 25, 68, 69, 52, 26, 35, 35, 27, 27,
        26, 39, 29, 25, 42, 35, 35, 36, 29, 27, 55, 53, 32, 25, 22, 24, 17, 17, 35, 30, 29, 32, 38, 44, 36, 36, 23, 19,
        22, 29, 26, 26
    ]

    # Load predictions
    data = np.load(preds_3d_file)
    if "coords3d_pred_world" not in data:
        raise KeyError(
            f"'coords3d_pred_world' not found in {preds_3d_file}. "
            f"Keys present: {list(data.keys())}"
        )
    poses3d = data["coords3d_pred_world"]

    cnt = 0
    cnt_activity = 0

    for subject in ("S9", "S11"):
        subject_dir = os.path.join(output_3d, subject)
        os.makedirs(subject_dir, exist_ok=True)
        activities = activities_S9 if subject == "S9" else activities_S11

        for activity in activities:
            act_len = activities_length[cnt_activity]
            activity_dir = os.path.join(subject_dir, activity)
            os.makedirs(activity_dir, exist_ok=True)

            # slice the 4-camera block for this activity
            preds_activity = poses3d[cnt : cnt + act_len * 4]
            print(f"3D Processing {subject} - {activity} with {len(preds_activity)} frames x 4 cams")

            for i, cam_name in enumerate(CAMERA_NAMES):
                cam_dir = os.path.join(activity_dir, cam_name)
                os.makedirs(cam_dir, exist_ok=True)

                start = i * act_len
                end   = (i + 1) * act_len
                poses3d_to_save = preds_activity[start:end, :, :]

                np.savez(os.path.join(cam_dir, "poses.npz"), poses3d=poses3d_to_save)

            cnt += act_len * 4
            cnt_activity += 1

    print("3D Done.")


def main():
    parser = argparse.ArgumentParser(description="Reorganize per-activity 2D and 3D predictions.")
    parser.add_argument("--input_dir", type=str, default="path/to/metrabs_preds_dir", help="Path to the per-activity 2D metrabs predictions directory.")
    parser.add_argument("--preds_3d", type=str, default="metrabs_3d_preds.npz", help="Path to the 3D metrabs prediction .npz file (expects key 'coords3d_pred_world').")
    parser.add_argument("--output_dir", type=str, default="../../data/h36m", help="Output dir where 2D/3D subfolders will be created.")
    args = parser.parse_args()

    preprocess_2d(args.input_dir, args.output_dir)
    preprocess_3d(args.preds_3d, args.output_dir)

if __name__ == "__main__":
    main()