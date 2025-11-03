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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter as gaussian_filter_np
import matplotlib.pyplot as plt



def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    # resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def unpack_covariance(cov_3d_compact):
    # Unpack the 6 values
    sigma_xx = cov_3d_compact[:, 0]
    sigma_xy = cov_3d_compact[:, 1]
    sigma_xz = cov_3d_compact[:, 2]
    sigma_yy = cov_3d_compact[:, 3]
    sigma_yz = cov_3d_compact[:, 4]
    sigma_zz = cov_3d_compact[:, 5]

    # Reconstruct the full covariance matrices using broadcasting
    cov_3d = torch.zeros((cov_3d_compact.size(0), 3, 3), device="cuda")
    cov_3d[:, 0, 0] = sigma_xx
    cov_3d[:, 0, 1] = sigma_xy
    cov_3d[:, 0, 2] = sigma_xz
    cov_3d[:, 1, 0] = sigma_xy
    cov_3d[:, 1, 1] = sigma_yy
    cov_3d[:, 1, 2] = sigma_yz
    cov_3d[:, 2, 0] = sigma_xz
    cov_3d[:, 2, 1] = sigma_yz
    cov_3d[:, 2, 2] = sigma_zz

    return cov_3d


# def get_2d_poses(poses_2d, std_dev_noise):

#     noise = torch.normal(0.0, std_dev_noise, poses_2d.shape).cuda()
#     n_poses_2d = poses_2d + noise
#     return n_poses_2d


def generate_heatmaps(gaussians, poses_2d, train_cameras, covariance_3d, dropout=False, data_root="data/h36m", nviews=4): 

    if "h36m" in data_root:
        n_joints = 17
    elif "panoptic" in data_root:
        n_joints = 19
    elif "occlusion-person" in data_root:
        n_joints = 15

    n_views = nviews

    heatmaps = TensorDict({})
    for i in range(n_views):
        heatmaps[str(i)] = torch.zeros((n_joints, train_cameras[i].image_height, train_cameras[i].image_width), device="cuda")

    view_matrix = torch.stack([camera.world_view_transform.T for camera in train_cameras])

    tan_fovx = torch.stack([
        torch.tan(torch.tensor(cam.FoVx * 0.5, device="cuda"))
        for cam in train_cameras
    ])

    tan_fovy = torch.stack([
        torch.tan(torch.tensor(cam.FoVy * 0.5, device="cuda"))
        for cam in train_cameras
    ])

    focal_x = torch.stack([
        train_cameras[i].image_width / (2.0 * tan_fovx[i])
        for i in range(n_views)
    ]).reshape(-1, 1)

    focal_y = torch.stack([
        train_cameras[i].image_height / (2.0 * tan_fovy[i])
        for i in range(n_views)
    ]).reshape(-1, 1)

    mean3D_hom = torch.cat([gaussians.get_xyz, torch.ones((gaussians.get_xyz.shape[0], 1), device="cuda")], dim=1)
    mean3D_trans = torch.matmul(view_matrix, mean3D_hom.T).transpose(1, 2)[:, :, :3]

    limx = (1.3 * tan_fovx).reshape(-1, 1)
    limy = (1.3 * tan_fovy).reshape(-1, 1)

    txtz = mean3D_trans[:, :, 0] / mean3D_trans[:, :, 2]
    tytz = mean3D_trans[:, :, 1] / mean3D_trans[:, :, 2]

    mean3D_trans[:, :, 0] = torch.clamp(txtz, -limx, limx) * mean3D_trans[:, :, 2]
    mean3D_trans[:, :, 1] = torch.clamp(tytz, -limy, limy) * mean3D_trans[:, :, 2]

    ja1 = (focal_x / mean3D_trans[:, :, 2]).unsqueeze(-1)
    ja2 = torch.zeros_like(ja1)
    ja3 = (- (focal_x * mean3D_trans[:, :, 0]) / mean3D_trans[:, :, 2] ** 2).unsqueeze(-1)
    ja123 = torch.cat([ja1, ja2, ja3], dim=2).unsqueeze(2)
    
    jb1 = torch.zeros_like(ja1)
    jb2 = (focal_y / mean3D_trans[:, :, 2]).unsqueeze(-1) 
    jb3 = (- (focal_y * mean3D_trans[:, :, 1]) / mean3D_trans[:, :, 2] ** 2).unsqueeze(-1)
    jb123 = torch.cat([jb1, jb2, jb3], dim=2).unsqueeze(2)

    jc1 = torch.zeros_like(ja1)
    jc2 = torch.zeros_like(ja1)
    jc3 = torch.zeros_like(ja1)
    jc123 = torch.cat([jc1, jc2, jc3], dim=2).unsqueeze(2)

    J = torch.cat([ja123, jb123, jc123], dim=2)  # 4 x 17 x 3 x 3
    W = view_matrix[:, :3, :3].unsqueeze(1)  # 4 x 1 x 3 x 3
    T = W @ J  # 4 x 17 x 3 x 3
    Vrk = covariance_3d

    T_T = T.permute(0, 1, 3, 2)
    Vrk_T = Vrk.permute(0, 2, 1)
    cov = T_T @ Vrk_T @ T

    cov_x = cov[:, :, 0, 0]
    cov_y = cov[:, :, 0, 1]
    cov_z = cov[:, :, 1, 1]

    h_var = 0.3
    cov_x += h_var
    cov_z += h_var
    det_cov_plus_h_cov = cov_x * cov_z - cov_y * cov_y  # Adjusted determinant

    # Invert covariance (EWA algorithm)
    det = det_cov_plus_h_cov

    mid = 0.5 * (cov_x + cov_z)
    lambda1 = mid + torch.sqrt(torch.max(torch.tensor(0.1), mid * mid - det))
    lambda2 = mid - torch.sqrt(torch.max(torch.tensor(0.1), mid * mid - det))
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(lambda1, lambda2)))
    lambda1 = torch.sqrt(lambda1)
    lambda2 = torch.sqrt(lambda2)

    dropout_cams = []
    if dropout:
        # generate 3 random numbers in range(4)
        dropout_cams = torch.randint(4, (3,))
        dropout_joints = torch.randint(n_joints, (3,))
     
    for i_cam in range(n_views):
        heatmaps_joints = heatmaps[str(i_cam)]
        x_coords = poses_2d[i_cam, :, 0].long()  # Shape: (num_points,)
        y_coords = poses_2d[i_cam, :, 1].long()  # Shape: (num_points,)
        x_coords = torch.clamp(x_coords, 0, heatmaps_joints.shape[2] - 1)
        y_coords = torch.clamp(y_coords, 0, heatmaps_joints.shape[1] - 1)

        joints_to_use = list(range(n_joints))
        if i_cam in dropout_cams:
            joints_to_use = list(filter(lambda x: x not in dropout_joints, joints_to_use))

        for j in joints_to_use:
            heatmaps_joints[j, y_coords[j], x_coords[j]] = 255
            heatmap = heatmaps_joints[j, ...]
            sigma1 = lambda1[i_cam, j].item()
            sigma2 = lambda2[i_cam, j].item()
            gaussian_heatmap = torch.as_tensor(gaussian_filter(cp.asarray(heatmap), sigma=[sigma1, sigma2]))
            heatmaps_joints[j, ...] = gaussian_heatmap
            

        
        heatmaps_joints = normalize_heatmaps(heatmaps_joints)
        heatmaps[str(i_cam)] = heatmaps_joints

    return heatmaps


def normalize_heatmaps(heatmaps):
    channel_min = heatmaps.view(heatmaps.size(0), -1).min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    channel_max = heatmaps.view(heatmaps.size(0), -1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    heatmaps = (heatmaps - channel_min) / (channel_max - channel_min + 1e-8)
    return heatmaps


# def compute_average_time(file_path):
#     try:
#         with open(file_path, 'r') as file:
#             values = [float(line.strip()) for line in file if line.strip()]
#             if not values:
#                 return 0
#             return sum(values) / len(values)
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#         return None
#     except ValueError:
#         print(f"Invalid value found in file: {file_path}")
#         return None
    

# def generate_heatmaps_channels(gaussians, poses_2d, bboxes, train_cameras, covariance_3d, dropout=False, data_root="data/h36m", nviews=4, channels=3): 

#     if "h36m" in data_root:
#         n_joints = 17
#     elif "panoptic" in data_root:
#         n_joints = 19
#     elif "occlusion-person" in data_root:
#         n_joints = 15

#     n_views = nviews

#     heatmaps = TensorDict({})
#     for i in range(n_views):
#         heatmaps[str(i)] = torch.zeros((n_joints, train_cameras[i].image_height, train_cameras[i].image_width), device="cuda")

#     view_matrix = torch.stack([camera.world_view_transform.T for camera in train_cameras])

#     tan_fovx = torch.stack([
#         torch.tan(torch.tensor(cam.FoVx * 0.5, device="cuda"))
#         for cam in train_cameras
#     ])

#     tan_fovy = torch.stack([
#         torch.tan(torch.tensor(cam.FoVy * 0.5, device="cuda"))
#         for cam in train_cameras
#     ])

#     focal_x = torch.stack([
#         train_cameras[i].image_width / (2.0 * tan_fovx[i])
#         for i in range(n_views)
#     ]).reshape(-1, 1)

#     focal_y = torch.stack([
#         train_cameras[i].image_height / (2.0 * tan_fovy[i])
#         for i in range(n_views)
#     ]).reshape(-1, 1)


#     mean3D_hom = torch.cat([gaussians.get_xyz, torch.ones((gaussians.get_xyz.shape[0], 1), device="cuda")], dim=1)
#     mean3D_trans = torch.matmul(view_matrix, mean3D_hom.T).transpose(1, 2)[:, :, :3]

#     limx = (1.3 * tan_fovx).reshape(-1, 1)
#     limy = (1.3 * tan_fovy).reshape(-1, 1)

#     txtz = mean3D_trans[:, :, 0] / mean3D_trans[:, :, 2]
#     tytz = mean3D_trans[:, :, 1] / mean3D_trans[:, :, 2]

#     mean3D_trans[:, :, 0] = torch.clamp(txtz, -limx, limx) * mean3D_trans[:, :, 2]
#     mean3D_trans[:, :, 1] = torch.clamp(tytz, -limy, limy) * mean3D_trans[:, :, 2]

#     ja1 = (focal_x / mean3D_trans[:, :, 2]).unsqueeze(-1)
#     ja2 = torch.zeros_like(ja1)
#     ja3 = (- (focal_x * mean3D_trans[:, :, 0]) / mean3D_trans[:, :, 2] ** 2).unsqueeze(-1)
#     ja123 = torch.cat([ja1, ja2, ja3], dim=2).unsqueeze(2)
    
#     jb1 = torch.zeros_like(ja1)
#     jb2 = (focal_y / mean3D_trans[:, :, 2]).unsqueeze(-1) 
#     jb3 = (- (focal_y * mean3D_trans[:, :, 1]) / mean3D_trans[:, :, 2] ** 2).unsqueeze(-1)
#     jb123 = torch.cat([jb1, jb2, jb3], dim=2).unsqueeze(2)

#     jc1 = torch.zeros_like(ja1)
#     jc2 = torch.zeros_like(ja1)
#     jc3 = torch.zeros_like(ja1)
#     jc123 = torch.cat([jc1, jc2, jc3], dim=2).unsqueeze(2)

#     J = torch.cat([ja123, jb123, jc123], dim=2)  # 4 x 17 x 3 x 3
#     W = view_matrix[:, :3, :3].unsqueeze(1)  # 4 x 1 x 3 x 3
#     T = W @ J  # 4 x 17 x 3 x 3
#     # Vrk = unpack_covariance(gaussians.get_covariance())
#     Vrk = covariance_3d

#     T_T = T.permute(0, 1, 3, 2)
#     Vrk_T = Vrk.permute(0, 2, 1)
#     cov = T_T @ Vrk_T @ T

#     cov_x = cov[:, :, 0, 0]
#     cov_y = cov[:, :, 0, 1]
#     cov_z = cov[:, :, 1, 1]

#     h_var = 0.3
#     cov_x += h_var
#     cov_z += h_var
#     det_cov_plus_h_cov = cov_x * cov_z - cov_y * cov_y  # Adjusted determinant

#     # Invert covariance (EWA algorithm)
#     det = det_cov_plus_h_cov

#     mid = 0.5 * (cov_x + cov_z)
#     lambda1 = mid + torch.sqrt(torch.max(torch.tensor(0.1), mid * mid - det))
#     lambda2 = mid - torch.sqrt(torch.max(torch.tensor(0.1), mid * mid - det))
#     radius = torch.ceil(3.0 * torch.sqrt(torch.max(lambda1, lambda2)))
#     lambda1 = torch.sqrt(lambda1)
#     lambda2 = torch.sqrt(lambda2)

#     for i_cam in range(n_views):
#         heatmaps_joints = heatmaps[str(i_cam)]
#         x_coords = poses_2d[i_cam, :, 0].long()  # Shape: (num_points,)
#         y_coords = poses_2d[i_cam, :, 1].long()  # Shape: (num_points,)
#         x_coords = torch.clamp(x_coords, 0, heatmaps_joints.shape[2] - 1)
#         y_coords = torch.clamp(y_coords, 0, heatmaps_joints.shape[1] - 1)

#         for j in list(range(n_joints)):
#             heatmaps_joints[j, y_coords[j], x_coords[j]] = 255
#             heatmap = heatmaps_joints[j, ...]
#             sigma1 = lambda1[i_cam, j].item()
#             sigma2 = lambda2[i_cam, j].item()
#             gaussian_heatmap = torch.as_tensor(gaussian_filter(cp.asarray(heatmap), sigma=[sigma1, sigma2]))
#             heatmaps_joints[j, ...] = gaussian_heatmap

#         heatmaps_joints = normalize_heatmaps(heatmaps_joints)
#         heatmaps[str(i_cam)] = heatmaps_joints

#     heatmaps_rgb = TensorDict({})
#     for i in range(n_views):
#         heatmaps_rgb[str(i)] = torch.zeros((channels, train_cameras[i].image_height, train_cameras[i].image_width), device="cuda")

#     for i_cam in range(n_views):
#         heatmaps_joints = heatmaps[str(i_cam)]
#         heatmap_rgb = heatmaps_rgb[str(i_cam)]
#         heatmap_rgb = torch.sum(heatmaps_joints, dim=0, keepdim=True).repeat(channels, 1, 1)
#         heatmap_rgb = normalize_heatmaps(heatmap_rgb)
#         heatmaps_rgb[str(i_cam)] = heatmap_rgb

#     return heatmaps_rgb


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, current_loss):
        # Check if loss has improved
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0  # Reset counter
        else:
            self.counter += 1  # Increase counter if no improvement

        # Stop if patience is exceeded
        return self.counter >= self.patience
    

class OptEarlyStopping:
    def __init__(self, window_size=4, repeat_tolerance=1e-6):

        self.window_size = window_size
        self.repeat_tolerance = repeat_tolerance
        self.loss_history = []

    def __call__(self, current_loss):
        self.loss_history.append(current_loss)

        total_len = len(self.loss_history)
        # Check if enough data to compare
        if total_len < 2 * self.window_size:
            return False 

        # Compare last two windows
        window1 = torch.tensor(self.loss_history[-2 * self.window_size : -self.window_size])
        window2 = torch.tensor(self.loss_history[-self.window_size :])

        # Measure similarity
        diff = torch.abs(window1 - window2)
        if torch.all(diff < self.repeat_tolerance):
            return True  # Pattern found

        return False
    
class NotStopping:
    def __init__(self):
        pass

    def __call__(self, current_loss):
        return False  # Never stop training