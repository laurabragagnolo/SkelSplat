from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import numpy as np

"""
Filter multi-view samples to those valid in *all* selected views, then save filtered 2D/3D predictions and GT with a suffix
`_filtered_{nviews}.npz`.
Valid views where metrabs produced a prediction (not None).


Directory layout expected (per activity):
- {data_path}/3d_metrabs_mono/S0/{activity}/{camera}/poses.npz          -> key: "poses"
- {data_path}/2d_metrabs/S0/{activity}/{camera}/poses.npz               -> key: "poses"
- {data_path}/2d_gt/S0/{activity}/{camera}/poses.npz                    -> key: "poses"
- {data_path}/3d_gt/S0/{activity}/poses.npz                             -> key: "poses"
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter samples valid across multiple views.")
    parser.add_argument("--data_path", type=Path, default="../../data/panoptic", help="Root dataset path.")
    parser.add_argument("--activities", type=str, nargs="+", default=["171204_pose5", "171204_pose6"], help="Activities to process.")
    parser.add_argument("--nviews", type=int, default=4, help="Number of views to enforce consistency on.")
    parser.add_argument("--cameras", type=str, nargs="*", default=["00_01", "00_02", "00_10", "00_13", "00_03", "00_23", "00_19", "00_30"],
        help="Ordered camera list to choose from (the first nviews will be used).")
    parser.add_argument("--preds3d_name", type=str, default="3d_metrabs_mono",help="Folder name for 3D mono predictions.")
    parser.add_argument("--preds2d_name", type=str, default="2d_metrabs", help="Folder name for 2D prediction inputs.")
    parser.add_argument("--gt2d_name", type=str, default="2d_gt", help="Folder name for 2D GT inputs.")
    parser.add_argument("--gt3d_name", type=str, default="3d_gt", help="Folder name for 3D GT inputs.")
    return parser.parse_args()


def load_npz(file, key):
    with np.load(file, allow_pickle=True) as npz:
        if key not in npz:
            raise KeyError(f"Key '{key}' not found in {file.name}")
        return npz[key]


def compute_valid_mask_across_views(view_arrays):
    """
    view_arrays: list of arrays with shape (N, ...)
    Returns a boolean mask of length N where a sample is True iff:
        - it exists (not None) in *every* view
        - contains no NaNs in any view's sample
    """
    if not view_arrays:
        raise ValueError("No view arrays provided.")
    lengths = [arr.shape[0] for arr in view_arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent number of samples across views: {lengths}")

    N = lengths[0]
    valid = np.ones(N, dtype=bool)

    # Handle dtype=object arrays (possible None entries).
    for v, arr in enumerate(view_arrays):
        if arr.dtype == object:
            mask = np.array([(x is not None) and (not np.any(np.isnan(x))) for x in arr], dtype=bool)
        else:
            # Numeric array: row is valid if row has no NaNs everywhere
            # (handles shapes like (N, J, D) by reducing over axes 1+)
            mask = ~np.isnan(arr).any(axis=tuple(range(1, arr.ndim)))
        valid &= mask

    return valid


def save_filtered(src_file, dst_file, indices, key):
    data = load_npz(src_file, key)
    filtered = np.asarray(data[indices], dtype=np.float64)
    if "gt" in dst_file:
        filtered = filtered * 10  # convert to cm
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    np.savez(dst_file, poses=filtered)
    return filtered.shape[0], filtered.shape


def main():
    args = parse_args()

    camera_names = args.cameras[:args.nviews]
    print(f"Using cameras: {camera_names}")

    for activity in args.activities:
        print(f"Activity: {activity} | views: {camera_names}")

        # 1) Load 3D mono predictions for each selected view to compute the validity mask
        view_pose_files = [
            os.path.join(args.data_path, args.preds3d_name, "S0", activity, cam, "poses.npz")
            for cam in camera_names
        ]
        try:
            preds_views = [load_npz(f, "poses") for f in view_pose_files]
        except (FileNotFoundError, KeyError) as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            continue

        N = preds_views[0].shape[0]
        print(f"Samples per view: {N}")

        valid_mask = compute_valid_mask_across_views(preds_views)
        valid_indices = np.nonzero(valid_mask)[0]
        print(f"Valid across all {len(camera_names)} views: {len(valid_indices)} / {N}")

        if len(valid_indices) == 0:
            print(f"[WARN] No valid samples for activity {activity}; skipping saves.", file=sys.stderr)
            continue

        # 2) Save filtered per-view 3D mono preds, 2D preds, and 2D GT
        for cam in camera_names:
            # 3D mono preds
            view_dir_3d = os.path.join(args.data_path, args.preds3d_name, "S0", activity, cam)
            src_3d = os.path.join(view_dir_3d, "poses.npz")
            dst_3d = os.path.join(view_dir_3d, f"poses_filtered_{len(camera_names)}.npz")
            try:
                cnt, shp = save_filtered(src_3d, dst_3d, valid_indices, "poses")
            except (FileNotFoundError, KeyError) as e:
                print(f"[ERROR] {e}", file=sys.stderr)

            # 2D preds
            view_dir_2d_preds = os.path.join(args.data_path, args.preds2d_name, "S0", activity, cam)
            src_2d_pred = os.path.join(view_dir_2d_preds, "poses.npz")
            dst_2d_pred = os.path.join(view_dir_2d_preds, f"poses_filtered_{len(camera_names)}.npz")
            try:
                cnt, shp = save_filtered(src_2d_pred, dst_2d_pred, valid_indices, "poses")
            except (FileNotFoundError, KeyError) as e:
                print(f"[ERROR] {e}", file=sys.stderr)

            # 2D GT
            view_dir_2d_gt = os.path.join(args.data_path, args.gt2d_name, "S0", activity, cam)
            src_2d_gt = os.path.join(view_dir_2d_gt, "poses.npz")
            dst_2d_gt = os.path.join(view_dir_2d_gt, f"poses_filtered_{len(camera_names)}.npz")
            try:
                cnt, shp = save_filtered(src_2d_gt, dst_2d_gt, valid_indices, "poses")
            except (FileNotFoundError, KeyError) as e:
                print(f"[ERROR] {e}", file=sys.stderr)

        # 3) Save filtered 3D GT (not per-view)
        view_dir_3d_gt = os.path.join(args.data_path, args.gt3d_name, "S0", activity)
        src_3d_gt = os.path.join(view_dir_3d_gt, "poses.npz")
        dst_3d_gt = os.path.join(view_dir_3d_gt, f"poses_filtered_{len(camera_names)}.npz")
        try:
            cnt, shp = save_filtered(src_3d_gt, dst_3d_gt, valid_indices, "poses")
        except (FileNotFoundError, KeyError) as e:
            print(f"[ERROR] {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    main()
