import copy
import json
import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from traitlets import default


def get_camera_params(path, cameras):
    with open(path) as f:
        calib_data = json.load(f)

    camera_parameters = {}

    # get the camera intrinsics and extrinsics
    for camera in cameras:
        # iterate through the cameras
        for params in calib_data["cameras"]:
            if params["name"] == camera:
                camera_intrinsics = params["K"]
                camera_rotation = params["R"]
                camera_translation = params["t"]

                camera_parameters[camera] = {"intrinsics": camera_intrinsics, "rotation": camera_rotation, "translation": camera_translation, "distortion": params["distCoef"]}
                break

    return camera_parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="path/to/your/panoptic-toolbox/")
    parser.add_argument("--sequences", nargs="+", type=str, default=["171204_pose5", "171204_pose6"])
    parser.add_argument("--output", type=str, default="../data/panoptic_prova")
    parser.add_argument("--cameras", nargs="+", type=int, default=["00_01", "00_02", "00_10", "00_13", "00_03", "00_19", "00_23", "00_30"])
    parser.add_argument("--visualize_2d", nargs="+", type=int, default=["00_01", "00_02", "00_10", "00_13"])
    args = parser.parse_args()

    for seq in args.sequences:

        outpath_3d_skeletons = os.path.join(args.output, "3d_gt")
        outpath_2d_skeletons = os.path.join(args.output, "2d_gt")

        # initialize the placeholders
        place_holder_2d = np.zeros((19, 3))
        place_holder_2d_list = []
        place_holder_3d = np.zeros((19, 4))
        place_holder_3d_list = []

        images_path = os.path.join(args.input, seq, "hdImgs")
        skeleton_path = os.path.join(args.input, seq, "hdPose3d_stage1_coco19")

        # read teh calibration file
        calib_file = os.path.join(args.input, seq, "calibration_{0}.json".format(seq))
        camera_parameters = get_camera_params(calib_file, args.cameras)

        poses_3d = []
        poses_2d = {"00_01": [], "00_02": [], "00_10": [], "00_13": [], "00_03": [], "00_19": [], "00_23": [], "00_30": []}
        boxes_2d = {"00_01": [], "00_02": [], "00_10": [], "00_13": [], "00_03": [], "00_19": [], "00_23": [], "00_30": []}

        # read the json files
        skeleton_files = os.listdir(skeleton_path)
        skeleton_files.sort()
        for k, file in enumerate(tqdm(skeleton_files)):

            # open the json file
            if file.endswith(".json"):
                try:
                    with open(os.path.join(skeleton_path, file)) as f:
                        data = json.load(f)
                except:
                    print("Error loading file: ", file)
                    continue

                if len(data["bodies"]) == 0:
                    print("No skeletons found in file: ", file)
                    continue
            else:
                continue

            # get the skeletons
            skeletons = data["bodies"]

            # iterate through the skeletons
            for skeleton in skeletons:

                # get the skeleton id and the 3d pose
                # skeleton_id = skeleton["id"]
                skeleton_joints = np.array(skeleton['joints19']).reshape(19,4)
                poses_3d.append(skeleton_joints[:, :3])

                # project the 3d points to 2d
                for camera in args.cameras:
                    camera_intrinsics = camera_parameters[camera]["intrinsics"]
                    camera_rotation = camera_parameters[camera]["rotation"]
                    camera_translation = camera_parameters[camera]["translation"]

                    # project the 3d points to 2d
                    skeleton_joints_3d = skeleton_joints[:, :3]
                    skeleton_joints_2d = np.dot(camera_intrinsics, np.dot(camera_rotation, skeleton_joints_3d.T) + camera_translation)
                    skeleton_joints_2d = skeleton_joints_2d[:2, :] / skeleton_joints_2d[2, :]
                    # add the confidence values equal to 1 (gt data)
                    # skeleton_joints_2d = np.vstack((skeleton_joints_2d, np.ones((1, 19))))
                    poses_2d[camera].append(skeleton_joints_2d.T)

        # save the 2d and 3d poses
        for camera in args.cameras:
            out_2d = os.path.join(args.output, "2d_gt", "S0", seq, camera)
            os.makedirs(out_2d, exist_ok=True)
            
            # save the 2d poses
            poses_2d_np = np.array(poses_2d[camera])
            print(poses_2d_np.shape)
            np.savez(os.path.join(out_2d, "poses.npz"), poses=poses_2d_np)

        out_3d = os.path.join(args.output, "3d_gt", "S0", seq)
        os.makedirs(out_3d, exist_ok=True)
        print(np.array(poses_3d).shape)
        np.savez(os.path.join(out_3d, "poses.npz"), poses=np.array(poses_3d))




