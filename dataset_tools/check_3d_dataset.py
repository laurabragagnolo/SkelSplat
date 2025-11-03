import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_poses_npz(npz_path):
    """Load 3D poses from a .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    try:
        return data['poses']
    except KeyError:
        return data['poses3d'] 

def plot_3d_poses(gt_poses, initial_guess_poses, subject, action, guess_gt=False, frame_step=64):
    """Plot 3D poses for visual comparison."""

    frames = [i_frame for i_frame in range(0, gt_poses.shape[0], frame_step)]
    gt_poses = gt_poses[frames, :, :]

    if guess_gt:
        initial_guess_poses = initial_guess_poses[frames, :, :]
    
    # Extracting 3D joint positions for the given frame
    for frame_idx in range(len(frames)):
        gt_pose = gt_poses[frame_idx, :, :]
        initial_guess_pose = initial_guess_poses[frame_idx, :, :]

        # Create figure and 3D axis
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the ground truth pose
        ax.scatter(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2], color='b', label='GT Pose', s=20)
        
        # Plot the initial guess pose
        ax.scatter(initial_guess_pose[:, 0], initial_guess_pose[:, 1], initial_guess_pose[:, 2], color='r', label='Initial Guess Pose', s=20)

        # Plot lines between the joints (can customize these based on your specific body parts)
        for i in range(0, gt_pose.shape[1]):  # Assuming there are 17 joints
            ax.plot([gt_pose[i, 0], initial_guess_pose[i, 0]],
                    [gt_pose[i, 1], initial_guess_pose[i, 1]],
                    [gt_pose[i, 2], initial_guess_pose[i, 2]], color='gray', linestyle='dotted', alpha=0.5)

        # Setting plot labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Comparison of 3D Poses: {subject} - {action} - Frame {frame_idx}")
        
        # Show legend
        ax.legend()

        # Show the plot
        plt.show()

def compare_3d_data_and_plot(gt_data_dir, initial_guess_dir):
    """Compare 3D GT data with Initial Guess (Metrabs) data and plot for visual comparison."""

    if "h36m" in gt_data_dir:
        frame_step = 64
    else:
        frame_step = 1
    
    # Iterate over each subject in the GT data directory
    for subject in sorted(os.listdir(gt_data_dir)):
        subject_path = os.path.join(gt_data_dir, subject)
        if not os.path.isdir(subject_path) or not subject.startswith("S"):
            continue  # Skip non-subject folders
        
        for action in sorted(os.listdir(subject_path)):
            action_path = os.path.join(subject_path, action)
            if not os.path.isdir(action_path):
                continue  # Skip non-action folders

            print(f"Visualizing {subject} - {action}")

            if "panoptic" in gt_data_dir:
                gt_data_file = os.path.join(action_path, "poses_filtered_8.npz")
                if "gt" in initial_guess_dir:
                    initial_guess_data_file = os.path.join(initial_guess_dir, subject, action, "poses_filtered_8.npz")
                else:
                    initial_guess_data_file = os.path.join(initial_guess_dir, subject, action, "poses.npz")
            else:   
                gt_data_file = os.path.join(action_path, "poses.npz")
                initial_guess_data_file = os.path.join(initial_guess_dir, subject, action, "poses.npz")

            if not os.path.exists(gt_data_file):
                print(f"GT file not found for {subject} - {action}")
                continue

            if not os.path.exists(initial_guess_data_file):
                print(f"Initial Guess file not found for {subject} - {action}")
                continue
            
            print("Comparing 3D poses for:", subject, action)
            # Load the 3D poses from both files
            gt_poses = load_poses_npz(gt_data_file)
            initial_guess_poses = load_poses_npz(initial_guess_data_file)

            # Plot the 3D poses for comparison (for a specific frame)
            plot_3d_poses(gt_poses, initial_guess_poses, subject, action, "gt" in initial_guess_data_file.lower(), frame_step)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Compare 3D GT data with Initial Guess data and plot.")
    parser.add_argument("--gt_data_dir", type=str, default="../data/occlusion-person/3d_gt", help="Path to the 3D GT data directory.")
    parser.add_argument("--initial_guess_dir", type=str, default="../data/occlusion-person/initial_guess/triang_resnet", help="Path to the Initial Guess data directory.")
    args = parser.parse_args()

    # Run the comparison and plotting
    compare_3d_data_and_plot(args.gt_data_dir, args.initial_guess_dir)

if __name__ == "__main__":
    main()
