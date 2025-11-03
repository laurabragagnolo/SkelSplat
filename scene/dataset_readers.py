#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from collections import defaultdict
import os
import sys
from PIL import Image
from typing import NamedTuple
import torch
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
# import transforms3d
import xml.etree.ElementTree as ET
# import re
# from scipy.ndimage import gaussian_filter
# import spacepy.pycdf
from scipy.spatial.transform import Rotation as Rotation
# import cv2

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
# from utils.viz_utils import show_joints_htmp, show_single_htmp, plot_rendering, save_rendering

import matplotlib.pyplot as plt


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    K: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    heatmap: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    scene_name: str = ""
    poses_2d: np.array = None
    bboxes: np.array = None


H36M_camera_size = [
        [(1002, 1000), (1000, 1000), (1000, 1000), (1002, 1000)],
        [(1002, 1000), (1000, 1000), (1000, 1000), (1002, 1000)],
        [(1002, 1000), (1000, 1000), (1000, 1000), (1002, 1000)],
        [(1002, 1000), (1002, 1000), (1002, 1000), (1002, 1000)],
        [(1002, 1000), (1000, 1000), (1000, 1000), (1002, 1000)],
        [(1002, 1000), (1000, 1000), (1000, 1000), (1002, 1000)],
        [(1002, 1000), (1002, 1000), (1002, 1000), (1002, 1000)],
        [(1002, 1000), (1000, 1000), (1000, 1000), (1002, 1000)],
        [(1002, 1000), (1000, 1000), (1000, 1000), (1002, 1000)],
        [(1002, 1000), (1002, 1000), (1002, 1000), (1002, 1000)],
        [(1002, 1000), (1000, 1000), (1000, 1000), (1002, 1000)]
    ]



class DataLoader:
    def __init__(self, data_root, initial_guess_dir, poses_2d_dir, frame_step=64, start_id=0, end_id=2181, nviews=4):
        self.data_root = data_root
        self.initial_guess_dir = initial_guess_dir
        self.poses_2d_dir = poses_2d_dir
        self.frame_step = frame_step

        self.start_id = start_id
        self.end_id = end_id

        self.gt_3d_dir = os.path.join(self.data_root, "3d_gt")
        self.gt_2d_dir = os.path.join(self.data_root, "2d_gt")

        self.n_views = nviews
        print(f"Number of views: {self.n_views}")

        if "h36m" in self.data_root:
            metadata_path = os.path.join(self.data_root, "initial_guess", "cameras", "camera-parameters.json")
            with open(metadata_path, "r") as file:
                self.camera_data = json.load(file)
            
            self.n_joints = 17
            self.im_width = 1000
            self.im_height = 1000
            self.cameras = ["54138969", "55011271", "58860488", "60457274"]

        elif "panoptic" in self.data_root:
            self.n_joints = 19
            self.im_width = 1920
            self.im_height = 1080
            self.cameras = ["00_01", "00_02", "00_10", "00_13", "00_03", "00_23", "00_19", "00_30"]

        elif "occlusion-person" in self.data_root:
            metadata_path = os.path.join(self.data_root, "cameras.json")
            with open(metadata_path, "r") as file:
                self.camera_data = json.load(file)
            
            self.n_joints = 15
            self.im_width = 1280
            self.im_height = 720
            self.cameras = ["0", "1", "2", "3", "4", "5", "6", "7"]

        self.scene_mapping = self.create_scene_mapping()

    
    def create_scene_mapping(self):
        scene_mapping = {}
        scene_id = 0
        
        subjects = sorted(os.listdir(self.initial_guess_dir))
        for subject in subjects:
            subject_path_3d = os.path.join(self.initial_guess_dir, subject)
            subject_path_2d = os.path.join(self.poses_2d_dir, subject)
            
            activities = sorted(os.listdir(subject_path_3d))
            for activity in activities:
                activity_path_3d = os.path.join(subject_path_3d, activity)
                activity_path_2d = os.path.join(subject_path_2d, activity)
                gt_3d_path = os.path.join(self.gt_3d_dir, subject, activity)

                print(f"Processing subject {subject}, activity {activity}")

                # Load 3D ground-truth poses
                poses_3d_gt = self.load_npz(os.path.join(gt_3d_path, "poses.npz"))
                if "panoptic" in self.data_root:
                    poses_3d_gt = self.load_npz(os.path.join(gt_3d_path, f"poses_filtered_{self.n_views}.npz"))

                poses_3d_gt = np.array([poses_3d_gt[i, ...] for i in range(0, poses_3d_gt.shape[0], self.frame_step)])

                # Load 3D initial guess poses
                if "gt" in self.initial_guess_dir:
                    poses_3d = poses_3d_gt
                else: 
                    poses_3d = self.load_npz(os.path.join(activity_path_3d, "poses.npz"))

                if not os.path.isdir(activity_path_2d):
                    print(f"Activity path {activity_path_2d} does not exist for subject {subject}, activity {activity}. Skipping...")
                    continue

                # Select nviews cameras
                cameras = self.cameras[:self.n_views]
                if "occlusion-person" in self.data_root and self.n_views == 4:
                    cameras = sorted(os.listdir(activity_path_2d))[1::2]
                    print(f"Selected cameras for occlusion-person: {cameras}")

                # Load 2D poses
                poses_2d_fcam = []
                for camera in cameras:

                    camera_path_2d = os.path.join(activity_path_2d, camera)
                    poses_2d = self.load_npz(os.path.join(camera_path_2d, "poses.npz"))[..., :2]
                    if "panoptic" in self.data_root:
                        poses_2d = self.load_npz(os.path.join(camera_path_2d, f"poses_filtered_{self.n_views}.npz"))[..., :2]
                    if "gt" in self.poses_2d_dir:
                        poses_2d = np.array([poses_2d[i, ...] for i in range(0, poses_2d.shape[0], self.frame_step)])[..., :2]

                    if poses_2d.shape[0] > poses_3d.shape[0]:
                        poses_2d = poses_2d[:poses_3d.shape[0], ...]
                    
                    poses_2d_fcam.append(poses_2d)
                
                poses_2d_fcam = np.array(poses_2d_fcam).reshape(self.n_views, -1, self.n_joints, 2)

                # Group information for each scene (frame index)
                for frame in range(poses_3d.shape[0]):
                    if self.end_id is not None and self.end_id > 0:
                        if scene_id >= self.end_id:
                            return scene_mapping  # Stop once we reach the desired limit


                    if scene_id >= self.start_id:
                        pose_3d_scene = poses_3d[frame, ...]
                        pose_3d_gt = poses_3d_gt[frame, ...]
                        poses_2d_scene = poses_2d_fcam[:, frame, :, :]

                        poses_2d_scene = torch.tensor(poses_2d_scene, device="cuda")
                        camera_info_fcam = []

                        for camera in cameras:
                            if "h36m" in self.data_root:
                                camera_info = getHuman36MCamera(self.camera_data, subject, camera)
                            elif "panoptic" in self.data_root:
                                camera_info = getPanopticCamera(self.data_root, activity, camera)
                            elif "occlusion-person" in self.data_root:
                                camera_info = getOcclusionPersonCamera(self.camera_data, scene_id, int(camera))
                            camera_info_fcam.append(camera_info)

                        cameras_scene = camera_info_fcam
                        frame_id = frame * self.frame_step
                        scene_name = f"{subject}_{activity}_{frame_id:06d}"

                        scene_mapping[scene_id] = (pose_3d_scene, pose_3d_gt, poses_2d_scene, cameras_scene, scene_name)

                    scene_id += 1

        return scene_mapping
    

    def load_npz(self, file_path):
        """Loads an NPZ file and extracts the first matching key."""
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            for key in ["poses", "poses2d", "boxes", "poses3d", "scores", "joint_errors"]:
                if key in data:
                    return data[key]
        return None
    
    
    def __len__(self):
        return len(self.scene_mapping)
        

    def __iter__(self):
        for scene_id, (pose_3d, pose_3d_gt, poses_2d, cameras, scene_name) in self.scene_mapping.items():
            yield scene_id, (pose_3d, pose_3d_gt, poses_2d, cameras, scene_name)



def getHuman36MCamera(data, subject, camera):
    
    camera_mapping = {"54138969": 0, "55011271": 1, "58860488": 2, "60457274": 3}

    calibration_matrix = data["intrinsics"].get(camera, {}).get("calibration_matrix", None)
    calibration_matrix = np.array(calibration_matrix).reshape(3, 3)

    extrinsics = data["extrinsics"].get(subject, {}).get(camera, {})
    rotation_matrix = extrinsics.get("R", None)
    translation_vector = extrinsics.get("t", None)

    x1 = 0
    y1 = 0
    subject_id = int(subject.strip("S")) - 1
    width, height = H36M_camera_size[subject_id][camera_mapping[camera]]
    heatmap = np.zeros((width, height))

    calibration_matrix[0, 2] -= x1
    calibration_matrix[1, 2] -= y1

    uid = camera_mapping[camera]
    R = np.array(rotation_matrix).reshape(3, 3)
    qvec = Rotation.from_matrix(R).as_quat()
    qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])
    R = np.transpose(qvec2rotmat(qvec))
    T = np.array(translation_vector).reshape(3,)

    focal_length_x = calibration_matrix[0][0]
    focal_length_y = calibration_matrix[1][1]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)
    
    image_path = ""
    depth_params = None
    image_name = ""
    depth_path = ""

    cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, K=calibration_matrix, depth_params=depth_params,
                            image_path=image_path, image_name=image_name, depth_path=depth_path,
                            width=width, height=height, heatmap=heatmap)

    return cam_info


def getPanopticCamera(data_dir, activity, camera):

    camera_mapping = {"00_01": 0, "00_02": 1, "00_10": 2, "00_13": 3, "00_03": 4, "00_23": 5, "00_19": 6, "00_30": 7}
    camera_names = ["00_01", "00_02", "00_10", "00_13", "00_03", "00_23", "00_19", "00_30"]

    metadata_path = os.path.join(data_dir, "3d_gt", "cameras", f"calibration_{activity}.json")
    with open(metadata_path, "r") as file:
        camera_data = json.load(file)

    for cam in camera_names:
        for data in camera_data["cameras"]:
            if data["name"] == camera:
                calibration_matrix = np.array(data["K"]).reshape(3, 3)
                rotation_matrix = np.array(data["R"]).reshape(3, 3)
                translation_vector = np.array(data["t"]).reshape(3, 1) * 10 # from cm to mm
                break

    x1 = 0
    y1 = 0
    width, height = (1920, 1080)  # Panoptic dataset camera resolution
    heatmap = np.zeros((width, height))

    calibration_matrix[0, 2] -= x1
    calibration_matrix[1, 2] -= y1

    uid = camera_mapping[camera]
    
    R = np.array(rotation_matrix).reshape(3, 3)
    qvec = Rotation.from_matrix(R).as_quat()
    qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])
    R = np.transpose(qvec2rotmat(qvec))
    T = np.array(translation_vector).reshape(3,)

    focal_length_x = calibration_matrix[0][0]
    focal_length_y = calibration_matrix[1][1]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)
    
    image_path = ""
    depth_params = None
    image_name = ""
    depth_path = ""
    
    cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, K=calibration_matrix, depth_params=depth_params,
                            image_path=image_path, image_name=image_name, depth_path=depth_path,
                            width=width, height=height, heatmap=heatmap)

    return cam_info


def getOcclusionPersonCamera(camera_data, scene_id, cam):
    
    cameras_scene = camera_data[str(scene_id)]
    camera = cameras_scene[cam]

    fx, fy = camera["fx"], camera["fy"]
    cx, cy = camera["cx"], camera["cy"]
    calibration_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
    rotation_matrix = np.array(camera["R"]).reshape(3, 3)
    translation_vector = np.array(camera["T"]).reshape(3, 1)
    translation_vector = -rotation_matrix @ translation_vector

    width, height = (1280, 720)  # camera resolution for occlusion-person dataset
    heatmap = np.zeros((width, height))

    x1 = 0
    y1 = 0
    calibration_matrix[0, 2] -= x1
    calibration_matrix[1, 2] -= y1

    uid = cam
    
    R = np.array(rotation_matrix).reshape(3, 3)
    R = np.transpose(R)
    T = np.array(translation_vector).reshape(3,)

    focal_length_x = calibration_matrix[0][0]
    focal_length_y = calibration_matrix[1][1]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)
    
    image_path = ""
    depth_params = None
    image_name = ""
    depth_path = ""
    
    cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, K=calibration_matrix, depth_params=depth_params,
                            image_path=image_path, image_name=image_name, depth_path=depth_path,
                            width=width, height=height, heatmap=heatmap)

    return cam_info



def readHuman36MSceneInfo(path, pose_3d, cameras, scene_name):

    ply_name = "points3D.ply"
    ply_path = os.path.join(path, "sparse", ply_name)
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    xyz = pose_3d.reshape(-1 , 3)
    rgb = np.ones_like(xyz) * 255       

    storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
    except:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        pcd = None

    # get cameras
    train_cam_infos = cameras
    test_cam_infos = []
    test_cam_names_list = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False,
                           scene_name=scene_name)
    return scene_info


def readPanopticSceneInfo(path, pose_3d, cameras, scene_name):

    ply_name = "points3D.ply"
    ply_path = os.path.join(path, "sparse", ply_name)
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    xyz = pose_3d.reshape(-1 , 3)
    rgb = np.ones_like(xyz) * 255       

    storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
    except:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        pcd = None

    # get cameras
    train_cam_infos = cameras
    test_cam_infos = []
    test_cam_names_list = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False,
                           scene_name=scene_name)
    return scene_info



def readOcclusionPersonSceneInfo(path, pose_3d, cameras, scene_name):

    ply_name = "points3D.ply"
    ply_path = os.path.join(path, "sparse", ply_name)
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    xyz = pose_3d.reshape(-1 , 3)
    rgb = np.ones_like(xyz) * 255       

    storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
    except:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        pcd = None

    # get cameras
    train_cam_infos = cameras
    test_cam_infos = []
    test_cam_names_list = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False,
                           scene_name=scene_name)
    return scene_info


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Human36M": readHuman36MSceneInfo,
    "Panoptic": readPanopticSceneInfo,
    "Occlusion-Person": readOcclusionPersonSceneInfo,
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}