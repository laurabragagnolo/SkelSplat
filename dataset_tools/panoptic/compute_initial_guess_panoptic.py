import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

def plot_pose(pose, skeleton, color, label=None):
    for joint_start, joint_end in skeleton:
        x = [pose[joint_start, 0], pose[joint_end, 0]]
        y = [pose[joint_start, 1], pose[joint_end, 1]]
        plt.plot(x, y, marker='o', markersize=4, color=color, alpha=0.8)
    plt.scatter(pose[:, 0], pose[:, 1], color=color, label=label)


def compute_reprojection_error(poses3d_world, poses2d, projection_matrices):

    nposes3d = ncams = poses3d_world.shape[0]
    nframes = poses3d_world.shape[1]
    njoints = poses3d_world.shape[2]

    projections2d = dict()
    overall_reprojerr = dict()

    # Convert world poses to homogeneous coordinates
    ones = np.ones((ncams, nframes, njoints, 1))
    poses3d_world_homo = np.concatenate([poses3d_world, ones], axis=-1)

    reprojection_errors_all = []

    for frame in range(nframes):
        for i_pose in range(nposes3d):
            projections2d[i_pose] = []
            overall_reprojerr[i_pose] = []
            l2_norms = []

            for i_image in range(ncams):
                trans3d = (projection_matrices[i_image] @ poses3d_world_homo[i_pose, frame].T).T 
                pose2d = trans3d[:, :2] / trans3d[:, 2:3]  # shape: (njoints, 2)
                
                # plt.figure(figsize=(6,8))
                # plot_pose(pose2d, skeleton, color='blue', label='Pose 1')
                # plot_pose(poses2d[i_image, frame], skeleton, color='green', label='Pose 2')

                # plt.gca().invert_yaxis()
                # plt.axis('equal')
                # plt.title("2D Human Poses")
                # plt.legend()
                # plt.show()

                projections2d[i_pose].append(pose2d)

            projections2d[i_pose] = np.stack(projections2d[i_pose], axis=0)  # shape: (ncams, njoints, 2)

            # Compute reprojection error (L2 norm) between reprojected and detected 2D poses
            for i_cam in range(ncams):
                pose2d_reproj = projections2d[i_pose][i_cam]      # shape: (njoints, 2)
                pose2d_det = poses2d[i_cam, frame]                # shape: (njoints, 2)
                diff = pose2d_reproj - pose2d_det
                diff = np.asarray(diff, dtype=np.float32)  # Ensure float32 for consistency
                l2_norm = np.linalg.norm(diff, axis=-1)           # shape: (njoints,)
                l2_norms.append(l2_norm)

            l2_norms = np.stack(l2_norms, axis=0)                  # shape: (ncams, njoints)
            mean_l2 = np.mean(l2_norms, axis=0).reshape(njoints, 1)  # shape: (njoints, 1)
            overall_reprojerr[i_pose] = mean_l2

        # shape: (ncams, njoints, 1)
        reprojection_errors = np.stack([overall_reprojerr[i] for i in range(ncams)], axis=0)
        reprojection_errors_all.append(reprojection_errors)

    # Final shape: (nframes, ncams, njoints)
    return np.array(reprojection_errors_all).squeeze(-1)


def convert_errors_to_weights(errors):

    weights = 1 / np.array(errors)  # Invert errors to get weights
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    return weights

def weighted_average_3d_joints(joints, weights):

    joints_array = np.array(joints)
    weighted_avg = np.average(joints_array, axis=0, weights=weights)
    return weighted_avg

def compute_weighted_average_pose(world_poses, poses2d, projection_matrices):
    avg_poses_all = []
    ncams = world_poses.shape[0]
    nframes = world_poses.shape[1]
    njoints = world_poses.shape[2]

    reprojection_errors_all = compute_reprojection_error(world_poses, poses2d, projection_matrices)

    for frame in range(nframes):
        avg_pose = []
        for j in range(njoints):
            j_poses = []
            for cam in range(ncams):
                j_poses.append(world_poses[cam, frame, j, :])

            # weights = compute_weight(reprojection_errors_all[frame, :, j])
            weights = convert_errors_to_weights(reprojection_errors_all[frame, :, j])
            # mean = weighted_average(j_poses, weights)
            mean = weighted_average_3d_joints(j_poses, weights)
            avg_pose.append(mean)
        avg_poses_all.append(avg_pose)

    return np.array(avg_poses_all).reshape(nframes, njoints, 3)


def get_camera_parameters(camera_data):
    camera_names = ["00_01", "00_02", "00_10", "00_13"]
    K = {}
    R = {}
    t = {}

    for cam in camera_names:
        for data in camera_data["cameras"]:
            if data["name"] == cam:
                K[cam] = np.array(data["K"]).reshape(3, 3)
                R[cam] = np.array(data["R"]).reshape(3, 3)
                t[cam] = np.array(data["t"]).reshape(3, 1) * 10 # from cm to mm
    
    return K, R, t


def create_projection_matrix(K_dict, R_dict, t_dict):
    P = []
    for cam in sorted(K_dict.keys()):
        K = K_dict[cam]
        R = R_dict[cam]
        t = t_dict[cam]
        RT = np.hstack((R, t.reshape(-1, 1)))  # Combine rotation and translation
        P.append(np.dot(K, RT))  # Projection matrix = K * [R | t]
    return P


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="../../data/panoptic", help="Path to the CMU panoptic dataset directory.")
    parser.add_argument("--preds_3d", type=str, default="3d_metrabs_mono", help="Path to the metrabs prediction file.")
    parser.add_argument("--preds_2d", type=str, default="2d_metrabs", help="Path to save the reorganized dataset.")
    parser.add_argument("--output_name", type=str, default="initial_guess/metrabs", help="Path to save the reorganized dataset.")
    args = parser.parse_args()

    # Define new directory structures
    output_root = os.path.join(args.root_dir, args.output_name)
    os.makedirs(output_root, exist_ok=True)

    camera_data = {}
    activities = ["171204_pose5", "171204_pose6"]

    for activity in activities:
        metadata_path = os.path.join(args.root_dir, "3d_gt", "cameras", f"calibration_{activity}.json")
        with open(metadata_path, "r") as file:
            camera_data[activity] = json.load(file)

    for subject in os.listdir(os.path.join(args.root_dir, args.preds_3d)):
        subject_path = os.path.join(args.root_dir, args.preds_3d, subject)
        subject_output_path = os.path.join(output_root, subject)
        os.makedirs(subject_output_path, exist_ok=True)

        for activity in sorted(os.listdir(subject_path)):
            activity_path = os.path.join(subject_path, activity)
            activity_output_path = os.path.join(subject_output_path, activity)
            os.makedirs(activity_output_path, exist_ok=True)
            print(activity_output_path)

            if not os.path.isdir(activity_path):
                continue

            preds_3d_fcam = []
            preds_2d_fcam = []
            
            for cam_name in sorted(os.listdir(activity_path)):
                cam_path_3d = os.path.join(activity_path, cam_name)
                cam_path_2d = os.path.join(args.root_dir, args.preds_2d, subject, activity, cam_name)

                if not os.path.isdir(cam_path_3d):
                    continue

                # Load 3D predictions
                preds_3d_path = os.path.join(cam_path_3d, "poses_filtered.npz")
                if not os.path.exists(preds_3d_path):
                    continue
                preds_3d_data = np.load(preds_3d_path, allow_pickle=True)
                poses3d = preds_3d_data['poses']

                preds_3d_fcam.append(poses3d)

                # Load 2D predictions
                preds_2d_path = os.path.join(cam_path_2d, "poses_filtered.npz")
                if not os.path.exists(preds_2d_path):
                    continue
                preds_2d_data = np.load(preds_2d_path, allow_pickle=True)
                poses2d = preds_2d_data['poses']
                preds_2d_fcam.append(poses2d)

            preds_3d_stack = np.stack(preds_3d_fcam, axis=0)
            preds_2d_stack = np.stack(preds_2d_fcam, axis=0)
            # Load cameras
            K_cameras, R_cameras, t_cameras = get_camera_parameters(camera_data[activity])
            
            P = create_projection_matrix(K_cameras, R_cameras, t_cameras)

            fused_poses = compute_weighted_average_pose(preds_3d_stack, preds_2d_stack, P) 

            fused_poses_path = os.path.join(activity_output_path, "poses.npz")
            np.savez(fused_poses_path, poses3d=fused_poses)
            print(f"Processed {subject}/{activity} and saved fused poses to {fused_poses_path}")





