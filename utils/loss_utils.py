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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips
def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

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


# ---------------------------------------------------------------------------
# ODE-specific losses  (used by train_ode.py)
# ---------------------------------------------------------------------------

def flow_consistency_loss(flow_rendered: torch.Tensor, flow_gt: torch.Tensor) -> torch.Tensor:
    """
    Optical flow consistency loss.

    Penalises the L1 difference between the optical flow estimated from
    consecutive *rendered* frames and the flow estimated from consecutive
    *ground-truth* frames.  Both flow tensors should already be estimated
    externally (e.g. via RAFT or TV-L1) before being passed here.

    This corresponds to:
        L_flow = sum_t ||F_hat_{t->t+1} - F_{t->t+1}||_1

    as described in the framework document.

    Args:
        flow_rendered : (..., 2, H, W) optical flow from rendered images.
        flow_gt       : (..., 2, H, W) optical flow from ground-truth images.

    Returns:
        Scalar L1 loss.
    """
    return torch.abs(flow_rendered - flow_gt).mean()


def ode_trajectory_reg_loss(
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    lambda_vel: float = 1.0,
    lambda_acc: float = 0.1,
) -> torch.Tensor:
    """
    Trajectory smoothness regularisation computed from sampled velocity and
    acceleration tensors (obtained from GaussianModel.get_velocity_and_acceleration).

    Penalises high velocity and acceleration of Gaussian centres, encouraging
    smooth, physically plausible trajectories.

    Args:
        velocities    : (T-1, N, 3) per-Gaussian velocity across T-1 intervals.
        accelerations : (T-2, N, 3) per-Gaussian acceleration across T-2 intervals.
        lambda_vel    : weight for velocity term.
        lambda_acc    : weight for acceleration term.

    Returns:
        Scalar regularisation loss.
    """
    vel_loss = (velocities ** 2).mean()
    acc_loss = (accelerations ** 2).mean()
    return lambda_vel * vel_loss + lambda_acc * acc_loss


def jitter_heatmap(velocities: torch.Tensor) -> torch.Tensor:
    """
    Per-Gaussian jitter score used for visualisation.

    Jitter is defined as the standard deviation of velocity magnitude across
    time steps.  High jitter indicates unstable / oscillating trajectories.

    Args:
        velocities : (T-1, N, 3) velocity tensor from get_velocity_and_acceleration.

    Returns:
        jitter : (N,) per-Gaussian jitter score (unnormalised).
    """
    speed = velocities.norm(dim=-1)   # (T-1, N)
    return speed.std(dim=0)           # (N,)

