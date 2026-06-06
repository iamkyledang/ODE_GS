"""
Deformable HexPlane + MLP Baseline for Dynamic 3D Gaussian Splatting.

Based on:
  "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"
  Wu et al., CVPR 2024.
  https://github.com/hustvl/4DGaussians

Architecture:
  - Multi-resolution HexPlane (6 planes from all pairs of (x,y,z,t)) for
    compact 4D space-time feature extraction.
  - Small MLP decoder mapping hexplane features →
    delta position (3), delta rotation (4), delta log-scale (3).
  - HexPlane resolution, channel dims, and multi-resolution multipliers are
    configurable via args (same fields as arguments/multipleview/default.py).

The HexPlaneField implementation is adapted from:
  https://github.com/hustvl/4DGaussians/blob/master/scene/hexplane.py
  (Apache-2.0 license)

Usage in full_eval.py:
  Pass --model_type deformable_hexplane_mlp to train.py / render.py.
"""

import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func, build_rotation,
    strip_symmetric, build_scaling_rotation,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


# ─────────────────────────────────────────────────────────────────────────────
# TV / smoothness helpers (adapted from hustvl/4DGaussians/scene/regulation.py)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_plane_tv(t: torch.Tensor) -> torch.Tensor:
    """Total variation loss for a 2-D plane grid of shape (1, C, H, W)."""
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w - 1]).sum()
    return 2.0 * (h_tv / count_h + w_tv / count_w)


def _compute_plane_smoothness(t: torch.Tensor) -> torch.Tensor:
    """2nd-order temporal smoothness loss (second difference along dim -2)."""
    batch_size, c, h, w = t.shape
    first_diff  = t[..., 1:, :] - t[..., :h - 1, :]           # (B, C, h-1, W)
    second_diff = first_diff[..., 1:, :] - first_diff[..., :h - 2, :]  # (B, C, h-2, W)
    return torch.square(second_diff).mean()


# ─────────────────────────────────────────────────────────────────────────────
# HexPlane field implementation (adapted from hustvl/4DGaussians)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_aabb(pts: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    """Normalise pts to [-1, 1]^D using the provided AABB."""
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def _grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor,
                          align_corners: bool = True) -> torch.Tensor:
    """Bilinear/trilinear grid sample with automatic batch dim insertion."""
    grid_dim = coords.shape[-1]
    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = F.grid_sample(grid, coords, align_corners=align_corners,
                            mode="bilinear", padding_mode="border")
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # (B, n, feature_dim)
    return interp.squeeze()


def _init_grid_param(grid_nd: int, in_dim: int, out_dim: int,
                     reso: list, a: float = 0.1, b: float = 0.5) -> nn.ParameterList:
    assert in_dim == len(reso)
    has_time = in_dim == 4
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        if has_time and 3 in coo_comb:
            nn.init.ones_(coef)
        else:
            nn.init.uniform_(coef, a=a, b=b)
        grid_coefs.append(coef)
    return grid_coefs


def _interpolate_ms_features(pts: torch.Tensor, ms_grids: list,
                              grid_dimensions: int, concat_features: bool,
                              num_levels=None) -> torch.Tensor:
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.0
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            feature_dim = grid[ci].shape[1]
            interp_out_plane = (
                _grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
            )
            interp_space = interp_space * interp_out_plane
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space
    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField(nn.Module):
    """
    Multi-resolution HexPlane feature field for 4D (x,y,z,t) space.

    Six 2D planes covering all pairs: (xy, xz, xt, yz, yt, zt).
    Features are multiplied across planes and concatenated across resolutions.
    """

    def __init__(self, bounds: float, planeconfig: dict, multires: list) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds],
                              [-bounds, -bounds, -bounds]], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            config["resolution"] = (
                [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            )
            gp = _init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        print(f"[HexPlaneField] feature_dim={self.feat_dim}")

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor(np.array([xyz_max, xyz_min]), dtype=torch.float32).cuda()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print("HexPlaneField: set aabb =", self.aabb)

    def forward(self, pts: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        pts = _normalize_aabb(pts, self.aabb)
        pts_t = torch.cat((pts, timestamps), dim=-1)
        pts_t = pts_t.reshape(-1, pts_t.shape[-1])
        features = _interpolate_ms_features(
            pts_t,
            ms_grids=self.grids,
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features,
            num_levels=None,
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=pts.device)
        return features


# ─────────────────────────────────────────────────────────────────────────────
# MLP decoder (small, following hustvl/4DGaussians Deformation class)
# ─────────────────────────────────────────────────────────────────────────────

def _init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            init.zeros_(m.bias)


class _HexDecodeMLP(nn.Module):
    """Small 2-layer MLP decoder on top of hexplane features."""

    def __init__(self, feat_dim: int, W: int = 128, D: int = 1):
        super().__init__()
        layers: list = [nn.Linear(feat_dim, W)]
        for _ in range(D):
            layers += [nn.ReLU(), nn.Linear(W, W)]
        self.trunk = nn.Sequential(*layers)
        self.pos_head   = nn.Sequential(nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
        self.rot_head   = nn.Sequential(nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))
        self.scale_head = nn.Sequential(nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
        self.apply(_init_weights)

    def forward(self, feat: torch.Tensor) -> tuple:
        h = self.trunk(feat)
        return self.pos_head(h), self.rot_head(h), self.scale_head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility shim for scene/__init__.py set_aabb call
# ─────────────────────────────────────────────────────────────────────────────

class _HexDeformNet:
    """Wraps HexPlaneField so scene/__init__.py can call set_aabb on it."""
    def __init__(self, hexplane: HexPlaneField):
        self._hexplane = hexplane

    def set_aabb(self, xyz_max, xyz_min):
        self._hexplane.set_aabb(xyz_max, xyz_min)


class _HexDeformation:
    def __init__(self, hexplane: HexPlaneField):
        self.deformation_net = _HexDeformNet(hexplane)

    def set_aabb(self, xyz_max, xyz_min):
        self.deformation_net.set_aabb(xyz_max, xyz_min)


# ─────────────────────────────────────────────────────────────────────────────
# GaussianModel — HexPlane + MLP baseline
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_KPLANES = {
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 16,
    "resolution": [64, 64, 64, 150],
}
_DEFAULT_MULTIRES = [1, 2]
_DEFAULT_BOUNDS = 1.6


class GaussianModel:
    """
    Dynamic 3D Gaussian Model using HexPlane + MLP deformation.

    Canonical state: standard 3DGS parameters (_xyz, _scaling, etc.)
    Deformation: HexPlane features + MLP decoder → (Δxyz, Δquat, Δscale)

    The HexPlane encodes space-time features for all 6 pairs of (x,y,z,t).
    A small MLP decodes these features to per-Gaussian deformation deltas.
    """

    def setup_functions(self):
        def _build_cov(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_cov = L @ L.transpose(1, 2)
            return strip_symmetric(actual_cov)

        self.scaling_activation         = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation      = _build_cov
        self.opacity_activation         = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation        = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, args):
        self.active_sh_degree  = 0
        self.max_sh_degree     = sh_degree
        self._args             = args

        # Standard 3DGS canonical parameters
        self._xyz            = torch.empty(0)
        self._features_dc    = torch.empty(0)
        self._features_rest  = torch.empty(0)
        self._scaling        = torch.empty(0)
        self._rotation       = torch.empty(0)
        self._opacity        = torch.empty(0)
        self.max_radii2D     = torch.empty(0)
        self.xyz_gradient_accum  = torch.empty(0)
        self._deformation_accum  = torch.empty(0)
        self.denom           = torch.empty(0)
        self.optimizer       = None
        self.percent_dense   = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)

        # Build HexPlane and MLP from args
        bounds   = getattr(args, "bounds", _DEFAULT_BOUNDS)
        sp_res   = getattr(args, "hexplane_spatial_res", 64)
        t_res    = getattr(args, "hexplane_time_res",    150)
        feat_dim = getattr(args, "hexplane_feat_dim",    16)
        dec_W    = getattr(args, "hexplane_decode_W",    128)
        dec_D    = getattr(args, "hexplane_decode_D",    1)

        kplanes_cfg = getattr(args, "kplanes_config", None) or {
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": feat_dim,
            "resolution": [sp_res, sp_res, sp_res, t_res],
        }
        multires = getattr(args, "multires", _DEFAULT_MULTIRES)

        self._hexplane  = HexPlaneField(bounds, kplanes_cfg, multires).cuda()
        self._decode_mlp = _HexDecodeMLP(feat_dim=self._hexplane.feat_dim, W=dec_W, D=dec_D).cuda()

        # Compatibility shim: wire AABB propagation
        self._deformation = _HexDeformation(self._hexplane)

        self.setup_functions()

    # ---------------------------------------------------------------- properties

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # -------------------------------------------------------------- init from pcd

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points).astype(np.float32)).cuda()
        fused_color       = RGB2SH(torch.from_numpy(np.asarray(pcd.colors).astype(np.float32)).cuda())
        N = fused_point_cloud.shape[0]

        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        print(f"[HexPlaneMLP] Initialising {N} Gaussians")

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales    = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots      = torch.zeros((N, 4), device="cuda");  rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), dtype=torch.float, device="cuda"))

        self._xyz           = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc   = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling       = nn.Parameter(scales.requires_grad_(True))
        self._rotation      = nn.Parameter(rots.requires_grad_(True))
        self._opacity       = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D    = torch.zeros(N, device="cuda")
        self._deformation_table = torch.gt(torch.ones(N, device="cuda"), 0)

    # ---------------------------------------------------------------- deformation

    def apply_ode_deformation(self, tau: float):
        """
        Query HexPlane + MLP at time tau ∈ [-1, 1].

        Returns:
            xyz_t   : (N, 3)
            scale_t : (N, 3)
            rot_t   : (N, 4)
        """
        t_norm = torch.tensor([(tau + 1.0) / 2.0], dtype=torch.float32, device="cuda")
        t_col  = t_norm.expand(self._xyz.shape[0], 1)   # (N, 1)

        feat = self._hexplane(self._xyz.detach(), t_col)  # (N, feat_dim)
        d_xyz, d_rot, d_scale = self._decode_mlp(feat)

        return self._xyz + d_xyz, self._scaling + d_scale, self._rotation + d_rot

    def compute_ode_regulation(self) -> torch.Tensor:
        """
        Spatial/temporal TV regularisation following hustvl/4DGaussians.

        Weights are read from args stored in __init__:
          plane_tv_weight        : TV loss on spatial planes (xy, xz, yz)
          time_smoothness_weight : 2nd-order smoothness on temporal planes (xt, yt, zt)
          l1_time_planes         : L1 penalty on temporal planes (encourages init values)

        Grid index mapping for 6 planes from itertools.combinations(range(4), 2):
          0=(0,1)=xy  1=(0,2)=xz  2=(0,3)=xt  3=(1,2)=yz  4=(1,3)=yt  5=(2,3)=zt
        """
        plane_tv_w = getattr(self._args, "plane_tv_weight",        0.0)
        time_sm_w  = getattr(self._args, "time_smoothness_weight",  0.0)
        l1_time_w  = getattr(self._args, "l1_time_planes",          0.0)

        if plane_tv_w == 0.0 and time_sm_w == 0.0 and l1_time_w == 0.0:
            return torch.tensor(0.0, device="cuda", requires_grad=True)

        # Spatial planes: (xy)=0, (xz)=1, (yz)=3
        # Temporal planes: (xt)=2, (yt)=4, (zt)=5
        spatial_ids  = [0, 1, 3]
        temporal_ids = [2, 4, 5]

        total = torch.tensor(0.0, device="cuda")

        for scale_grids in self._hexplane.grids:
            n_planes = len(scale_grids)
            for gid in spatial_ids:
                if gid < n_planes and plane_tv_w != 0.0:
                    g = scale_grids[gid]   # (1, C, H, W)
                    total = total + plane_tv_w * _compute_plane_tv(g)
            for gid in temporal_ids:
                if gid < n_planes:
                    g = scale_grids[gid]
                    if time_sm_w != 0.0:
                        total = total + time_sm_w * _compute_plane_smoothness(g)
                    if l1_time_w != 0.0:
                        # temporal planes are initialised to 1.0; penalise deviation
                        total = total + l1_time_w * torch.abs(1.0 - g).mean()

        return total

    # ---------------------------------------------------------------- training setup

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        N = self.get_xyz.shape[0]

        self.xyz_gradient_accum  = torch.zeros((N, 1), device="cuda")
        self._deformation_accum  = torch.zeros((N, 3), device="cuda")
        self.denom               = torch.zeros((N, 1), device="cuda")

        grid_lr   = getattr(training_args, "grid_lr",   2e-3)
        deform_lr = getattr(training_args, "deform_lr", 2e-4)

        grid_params   = list(self._hexplane.parameters())
        decode_params = list(self._decode_mlp.parameters())

        l = [
            {"params": [self._xyz],           "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc],   "lr": training_args.feature_lr,                               "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0,                        "name": "f_rest"},
            {"params": [self._opacity],       "lr": training_args.opacity_lr,                               "name": "opacity"},
            {"params": [self._scaling],       "lr": training_args.scaling_lr,                               "name": "scaling"},
            {"params": [self._rotation],      "lr": training_args.rotation_lr,                              "name": "rotation"},
            {"params": grid_params,           "lr": grid_lr,                                                 "name": "hexplane"},
            {"params": decode_params,         "lr": deform_lr,                                               "name": "decode_mlp"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self.xyz_scheduler_args(iteration)

    # ---------------------------------------------------------------- densification

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation, new_deformation_table):
        d = {
            "xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest,
            "opacity": new_opacities, "scaling": new_scaling, "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz           = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]
        self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], -1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom              = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D        = torch.zeros(self.get_xyz.shape[0],     device="cuda")

    def prune_points(self, mask):
        valid = ~mask
        optimizable_tensors = self._prune_optimizer(valid)
        self._xyz           = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]
        self._deformation_accum  = self._deformation_accum[valid]
        self.xyz_gradient_accum  = self.xyz_gradient_accum[valid]
        self._deformation_table  = self._deformation_table[valid]
        self.denom               = self.denom[valid]
        self.max_radii2D         = self.max_radii2D[valid]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad   = torch.zeros(n_init_points, device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        if not selected_pts_mask.any():
            return
        stds    = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means   = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots    = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                  self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling     = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation    = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity     = self._opacity[selected_pts_mask].repeat(N, 1)
        new_deform_table = self._deformation_table[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacity, new_scaling, new_rotation, new_deform_table)
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent,
                          density_threshold=20, displacement_scale=20,
                          model_path=None, iteration=None, stage=None):
        selected_pts_mask = torch.logical_and(
            torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False),
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        self.densification_postfix(
            self._xyz[selected_pts_mask], self._features_dc[selected_pts_mask],
            self._features_rest[selected_pts_mask], self._opacity[selected_pts_mask],
            self._scaling[selected_pts_mask], self._rotation[selected_pts_mask],
            self._deformation_table[selected_pts_mask])

    def densify(self, max_grad, min_opacity, extent, max_screen_size,
                density_threshold=5, displacement_scale=5,
                model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale,
                               model_path, iteration, stage)
        self.densify_and_split(grads, max_grad, extent)

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # ---------------------------------------------------------------- optimizer utils

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"]    = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"]    = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            if group["name"] not in tensors_dict:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ---------------------------------------------------------------- save / load

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f"f_dc_{i}")
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f"f_rest_{i}")
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz       = self._xyz.detach().cpu().numpy()
        normals   = np.zeros_like(xyz)
        f_dc      = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest    = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale     = self._scaling.detach().cpu().numpy()
        rotation  = self._rotation.detach().cpu().numpy()
        dtype_full = [(attr, "f4") for attr in self.construct_list_of_attributes()]
        elements   = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        scale_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rot_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._xyz           = nn.Parameter(torch.from_numpy(xyz.astype(np.float32)).cuda().requires_grad_(True))
        self._features_dc   = nn.Parameter(torch.from_numpy(features_dc.astype(np.float32)).cuda().transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.from_numpy(features_extra.astype(np.float32)).cuda().transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity       = nn.Parameter(torch.from_numpy(opacities.astype(np.float32)).cuda().requires_grad_(True))
        self._scaling       = nn.Parameter(torch.from_numpy(scales.astype(np.float32)).cuda().requires_grad_(True))
        self._rotation      = nn.Parameter(torch.from_numpy(rots.astype(np.float32)).cuda().requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def save_deformation(self, path):
        state = {
            "hexplane": self._hexplane.state_dict(),
            "decode_mlp": self._decode_mlp.state_dict(),
        }
        torch.save(state, os.path.join(path, "hexplane_deformation.pth"))
        torch.save(self._deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, "deformation_accum.pth"))

    def load_model(self, path):
        ckpt_path = os.path.join(path, "hexplane_deformation.pth")
        if not os.path.exists(ckpt_path):
            print(f"[HexPlaneMLP] No hexplane_deformation.pth found at {path}")
            return
        state = torch.load(ckpt_path, map_location="cuda")
        self._hexplane.load_state_dict(state["hexplane"])
        self._decode_mlp.load_state_dict(state["decode_mlp"])
        self._hexplane  = self._hexplane.cuda()
        self._decode_mlp = self._decode_mlp.cuda()
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(
                os.path.join(path, "deformation_table.pth"), map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(
                os.path.join(path, "deformation_accum.pth"), map_location="cuda")
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device="cuda")

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            {"hexplane": self._hexplane.state_dict(), "decode_mlp": self._decode_mlp.state_dict()},
            self._deformation_table,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree, self._xyz, net_state, self._deformation_table,
            self._features_dc, self._features_rest, self._scaling, self._rotation,
            self._opacity, self.max_radii2D, xyz_gradient_accum, denom,
            opt_dict, self.spatial_lr_scale,
        ) = model_args
        self._hexplane.load_state_dict(net_state["hexplane"])
        self._decode_mlp.load_state_dict(net_state["decode_mlp"])
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom              = denom
        self.optimizer.load_state_dict(opt_dict)
