import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Function to load .npz files
def load_npz(file_path):
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
        try:
            return data["poses"]
        except KeyError:
            try:
                return data["poses2d"]
            except KeyError:
                    return data["poses3d"]
                
    return None

# Function to draw poses and boxes
def draw_annotations(image, pose, box, color):
    if pose is not None:
        for i in range(pose.shape[0]):
            x = int(pose[i, 0])
            y = int(pose[i, 1])
            cv2.circle(image, (x, y), 3, color, -1)  # Draw joint
            cv2.putText(image, str(i), (x + 5, y - 5),  # Add label near the joint
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    
    if box is not None:
        x1, y1, width, height = box[0], box[1], box[2], box[3]
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    return image


# Parse command line arguments
parser = argparse.ArgumentParser(description="Compare 2D pose and box annotations.")
parser.add_argument("--gt_dir", type=str, help="Path to ground truth directory.", default="../data/panoptic/2d_gt")
parser.add_argument("--pred_dir", type=str, help="Path to predicted annotations directory.", default="../data/panoptic/2d_metrabs")
args = parser.parse_args()

# Define paths
gt_root = args.gt_dir
pred_root = args.pred_dir

# Define colors
gt_color = (0, 255, 0)  # Green for GT
pred_color = (0, 0, 255)  # Red for predictions

# Define image size
if "h36m" in gt_root:
    img_width, img_height = 1000, 1000
    frame_step = 64
elif "occlusion-person" in gt_root:
    img_width, img_height = 1280, 720
    frame_step = 1
elif "panoptic" in gt_root:
    img_width, img_height = 1920, 1080
    frame_step = 1

subjects = os.listdir(gt_root)
for subject in sorted(subjects):
    activities = sorted(os.listdir(os.path.join(gt_root, subject)))
    for activity in activities:
        cameras = sorted(os.listdir(os.path.join(gt_root, subject, activity)))
        for camera in cameras:
            print(f"Visualizing {subject} - {activity} - {camera}")
            if "panoptic" in gt_root:
                gt_pose_path = os.path.join(gt_root, subject, activity, camera, "poses_filtered_8.npz")
                pred_pose_path = os.path.join(pred_root, subject, activity, camera, "poses_filtered_8.npz")
            else:
                gt_pose_path = os.path.join(gt_root, subject, activity, camera, "poses.npz")
                pred_pose_path = os.path.join(pred_root, subject, activity, camera, "poses.npz")
            
            gt_poses = load_npz(gt_pose_path)
            gt_poses_sampled = np.array([gt_poses[i, ...] for i in range(0, gt_poses.shape[0], frame_step)])
            pred_poses = load_npz(pred_pose_path)

            num_frames = len(gt_poses_sampled)
            for frame_id in range(num_frames):
            # for frame_id in [0]:
                frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                draw_annotations(frame, gt_poses_sampled[frame_id, ...], None, gt_color)
                draw_annotations(frame, pred_poses[frame_id, ...], None, pred_color)
                # Display the image
                cv2.imshow(f"{activity} - {camera} - Frame {frame_id}", frame)
                cv2.waitKey(0)  # Short pause to visualize
                cv2.destroyAllWindows()
        