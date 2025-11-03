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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass
import matplotlib.pyplot as plt 

C1 = 0.01 ** 2
C2 = 0.03 ** 2

# plt.ion()  # Turn on interactive mode
# fig, ax = plt.subplots(figsize=(8, 4))
# scatter_gt, = ax.plot([], [], 'bo', label='Ground Truth')  # Blue dots
# line_gt, = ax.plot([], [], 'b--', alpha=0.5)  # Dashed line for GT
# scatter_pred, = ax.plot([], [], 'ro', label='Predicted')  # Red dots
# line_pred, = ax.plot([], [], 'r--', alpha=0.5)  # Dashed line for predictions

# ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Reference line
# ax.set_xlabel("Index")
# ax.set_ylabel("Value")
# ax.set_title("1D Tensor Values")
# ax.legend()
# ax.grid(True)

def softargmax2d(inp, beta=100):

    *_, h, w = inp.shape  # Get height and width

    # Flatten the spatial dimensions
    inp = inp.view(*_, h * w)

    # Apply softmax over the spatial dimensions
    inp = nn.functional.softmax(beta * inp, dim=-1)

    # Create coordinate grids normalized to [0, 1]
    indices_r = torch.linspace(0, 1, steps=h, device=inp.device).view(-1, 1).repeat(1, w).view(1, h * w)
    indices_c = torch.linspace(0, 1, steps=w, device=inp.device).view(1, -1).repeat(h, 1).view(1, h * w)

    # Compute soft-argmax as weighted sums of coordinates
    result_r = torch.sum(inp * indices_r, dim=-1) * (h - 1)
    result_c = torch.sum(inp * indices_c, dim=-1) * (w - 1)

    # Combine row and column results into a single tensor
    result = torch.stack([result_c, result_r], dim=-1)

    del result_r, result_c, indices_r, indices_c

    return result


def l1_loss(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):
    loss = torch.abs((rendering - gt_heatmap))
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def l2_loss(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):
    predicted_pose = softargmax2d(rendering)
    loss = ((predicted_pose - gt_2d) ** 2)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def l2_loss_gaussian(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):

    gt_mask = gt_heatmap > 0
    rendering_mask = rendering > 0

    mask = gt_mask | rendering_mask

    error = ((rendering - gt_heatmap) ** 2)
    error_mean = error.mean()
    loss = error[mask]
    if reduction == 'mean':
        return loss.mean(), error
    elif reduction == 'sum':
        return loss.sum()
    return loss


def l1_loss_gaussian(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):

    gt_mask = gt_heatmap > 0
    rendering_mask = rendering > 0

    mask = gt_mask | rendering_mask

    error = torch.abs((rendering - gt_heatmap))
    error_mean = error.mean()
    loss = error[mask]
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def l2_loss_gaussian_l1_loss_gaussian(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):
    l2 = l2_loss_gaussian(rendering, gt_heatmap, gt_2d, lambda_loss, reduction='none')
    l1 = l1_loss_gaussian(rendering, gt_heatmap, gt_2d, lambda_loss, reduction='none')
    if reduction == 'mean':
        return (1.0 - lambda_loss) * l2.mean() + lambda_loss * l1.mean()
    elif reduction == 'sum':
        return (1.0 - lambda_loss) * l2.sum() + lambda_loss * l1.sum()
    return (1.0 - lambda_loss) * l2 + lambda_loss * l1



def l2_loss_sqrt(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):
    predicted_pose = softargmax2d(rendering)
    loss = torch.sqrt(((predicted_pose - gt_2d) ** 2).sum())
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def huber_loss(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, delta=1.0, reduction='mean'):
    predicted_pose = softargmax2d(rendering)
    error = torch.abs(predicted_pose - gt_2d)
    is_small_error = error <= delta
    loss = torch.where(is_small_error, error ** 2, torch.abs(delta - error) - 0.5 * delta)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def l1_l2_loss(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):
    l1 = l1_loss(rendering, gt_heatmap, gt_2d, lambda_loss, reduction='none')
    l2 = l2_loss(rendering, gt_heatmap, gt_2d, lambda_loss, reduction='none')
    if reduction == 'mean':
        return (1.0 - lambda_loss) * l1.mean() + lambda_loss * l2.mean()
    elif reduction == 'sum':
        return (1.0 - lambda_loss) * l1.sum() + lambda_loss * l2.sum()
    return (1.0 - lambda_loss) * l1 + lambda_loss * l2


def l1_huber_loss(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, delta=1.0, reduction='mean'):
    l1 = l1_loss(rendering, gt_heatmap, gt_2d, lambda_loss, reduction='none')
    huber = huber_loss(rendering, gt_heatmap, gt_2d, lambda_loss, delta, reduction='none')
    if reduction == 'mean':
        return (1.0 - lambda_loss) * l1.mean() + lambda_loss * huber.mean()
    elif reduction == 'sum':
        return (1.0 - lambda_loss) * l1.sum() + lambda_loss * huber.sum()
    return (1.0 - lambda_loss) * l1 + lambda_loss * huber


def l1_loss_masked(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):
    gt_mask = gt_heatmap > 0
    values = gt_heatmap[gt_mask].detach().cpu().numpy()
    values_pred = rendering[gt_mask].detach().cpu().numpy()
    rendering_mask = rendering > 0
    mask = gt_mask | rendering_mask

    # Create an index array for the x-axis
    indices = torch.arange(len(values))
    indices_pred = torch.arange(len(values_pred))

    # loss = torch.zeros_like(gt_heatmap)
    error = torch.abs(rendering - gt_heatmap)
    error_mean = error.mean()
    loss = error[mask]
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def l1_masked_l2_loss(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):
    l1_masked = l1_loss_masked(rendering, gt_heatmap, gt_2d, lambda_loss, reduction='none')
    l2 = l2_loss(rendering, gt_heatmap, gt_2d, lambda_loss, reduction='none')
    if reduction == 'mean':
        return (1.0 - lambda_loss) * l1_masked.mean() + lambda_loss * l2.mean()
    elif reduction == 'sum':
        return (1.0 - lambda_loss) * l1_masked.sum() + lambda_loss * l2.sum()
    return (1.0 - lambda_loss) * l1_masked + lambda_loss * l2


def l1_masked_huber_loss(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, delta=1.0, reduction='mean'):
    l1_masked = l1_loss_masked(rendering, gt_heatmap, gt_2d, lambda_loss, reduction='none')
    huber = huber_loss(rendering, gt_heatmap, gt_2d, lambda_loss, delta, reduction='none')
    if reduction == 'mean':
        return (1.0 - lambda_loss) * l1_masked.mean() + lambda_loss * huber.mean()
    elif reduction == 'sum':
        return (1.0 - lambda_loss) * l1_masked.sum() + lambda_loss * huber.sum()
    return (1.0 - lambda_loss) * l1_masked + lambda_loss * huber


def cauchy_loss(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):
    predicted_pose = softargmax2d(rendering)
    residual = predicted_pose - gt_2d
    loss = torch.log(1 + (residual / 1.0) ** 2)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def limb_3d_consistency_loss(gaussians_xyz, data_root, reduction="mean"):

    if "h36m" in data_root:

        l_arm = torch.norm(gaussians_xyz[12, ...] - gaussians_xyz[13, ...], dim=-1)
        r_arm = torch.norm(gaussians_xyz[15, ...] - gaussians_xyz[16, ...], dim=-1)
        l_leg = torch.norm(gaussians_xyz[5, ...] - gaussians_xyz[6, ...], dim=-1)
        r_leg = torch.norm(gaussians_xyz[2, ...] - gaussians_xyz[3, ...], dim=-1)

    elif "panoptic" in data_root:

        l_arm = torch.norm(gaussians_xyz[4, ...] - gaussians_xyz[5, ...], dim=-1)
        r_arm = torch.norm(gaussians_xyz[10, ...] - gaussians_xyz[11, ...], dim=-1)
        l_leg = torch.norm(gaussians_xyz[7, ...] - gaussians_xyz[8, ...], dim=-1)
        r_leg = torch.norm(gaussians_xyz[13, ...] - gaussians_xyz[14, ...], dim=-1)
    
    elif "occlusion-person" in data_root:

        l_arm = torch.norm(gaussians_xyz[10, ...] - gaussians_xyz[11, ...], dim=-1)
        r_arm = torch.norm(gaussians_xyz[13, ...] - gaussians_xyz[14, ...], dim=-1)
        l_leg = torch.norm(gaussians_xyz[5, ...] - gaussians_xyz[6, ...], dim=-1)
        r_leg = torch.norm(gaussians_xyz[2, ...] - gaussians_xyz[3, ...], dim=-1)

    loss_3d_consistency = torch.norm(l_arm - r_arm) + torch.norm(l_leg - r_leg)
    return loss_3d_consistency
    

def no_consistency(rendering, gt_heatmap, gt_2d, lambda_loss=1.0, reduction='mean'):
    return torch.tensor(0.0)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
