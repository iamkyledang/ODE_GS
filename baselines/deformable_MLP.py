"""
Deformable MLP Baseline for Dynamic 3D Gaussian Splatting.

Based on:
  "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction"
  Yang et al., CVPR 2024.
  https://github.com/ingra14m/Deformable-3D-Gaussians

Architecture:
  - Positional encoding (Fourier features) for 3D position and time.
  - 8-layer MLP (width 256) mapping (pos_enc(x), time_enc(t)) →
    delta position (3), delta rotation (4), delta log-scale (3).
  - Per-Gaussian deformation is computed each frame; no per-Gaussian parameters.
  - The network is shared across all Gaussians and frames.

Usage in full_eval.py:
  Pass --model_type deformable_mlp to train.py / render.py.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

# Ensure project root is on the import path
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
# Positional / temporal encoding (Fourier features)
# ─────────────────────────────────────────────────────────────────────────────

def poc_fre(x: torch.Tensor, freq_buf: torch.Tensor) -> torch.Tensor:
    """
    Positional encoding following ingra14m/Deformable-3D-Gaussians.

    x        : (N, D)
    freq_buf : (L,)  powers-of-two [2^0, ..., 2^{L-1}]
    returns  : (N, D + 2*D*L)  = [x, sin(x*f0), cos(x*f0), ..., sin(x*fL), cos(x*fL)]
    """
    xb = (x.unsqueeze(-1) * freq_buf).flatten(-2)      # (N, D*L)
    return torch.cat([x, xb.sin(), xb.cos()], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Deformation MLP (based on ingra14m's deform_network + Deformation classes)
# ─────────────────────────────────────────────────────────────────────────────

class _DeformMLP(nn.Module):
    """
    MLP deformation network.

    Input  : concatenation of pos_enc(xyz) and time_enc(t)
    Output : (Δxyz, Δquat, Δlog_scale)

    Architecture mirrors ingra14m's Deformation class with no_grid=False
    replaced by pure MLP (no_grid=True logic but with full positional encoding).
    """

    def __init__(self, pos_ch: int, t_ch: int, W: int = 256, D: int = 8):
        super().__init__()
        in_ch = pos_ch + t_ch
        layers: list = [nn.Linear(in_ch, W), nn.ReLU()]
        for _ in range(D - 1):
            layers += [nn.Linear(W, W), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.pos_head   = nn.Sequential(nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
        self.rot_head   = nn.Sequential(nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))
        self.scale_head = nn.Sequential(nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
        self.apply(_init_weights)

    def forward(self, pos_emb: torch.Tensor, t_emb: torch.Tensor) -> tuple:
        # t_emb: (1, t_ch) → broadcast to (N, t_ch)
        h = self.trunk(torch.cat([pos_emb, t_emb.expand(pos_emb.shape[0], -1)], dim=-1))
        return self.pos_head(h), self.rot_head(h), self.scale_head(h)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            init.zeros_(m.bias)


# ─────────────────────────────────────────────────────────────────────────────
# No-op compatibility shim for scene/__init__.py set_aabb call
# ─────────────────────────────────────────────────────────────────────────────

class _NoOpDeformNet:
    def set_aabb(self, xyz_max, xyz_min):
        pass


class _NoOpDeformation:
    def __init__(self):
        self.deformation_net = _NoOpDeformNet()

    def set_aabb(self, xyz_max, xyz_min):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# GaussianModel — Deformable MLP baseline
# ─────────────────────────────────────────────────────────────────────────────

class GaussianModel:
    """
    Dynamic 3D Gaussian Model with MLP-based deformation field.

    Canonical state (at tau = 0):
        _xyz, _scaling, _rotation, _opacity, _features_dc, _features_rest

    Shared deformation network:
        _deform_net : MLP mapping (pos_enc(xyz), time_enc(t)) -> (Dxyz, Dquat, Dscale)

    Architecture (overridable via args / full_eval.py METHOD_CONFIGS):
        deform_mlp_width : hidden layer width  (default 256)
        deform_mlp_depth : number of hidden layers (default 8)
        deform_pos_pe    : positional encoding freqs for xyz (default 10)
        deform_time_pe   : positional encoding freqs for t   (default 4)
    """

    # Fallback defaults (used when args does not carry the field)
    _POS_BASE_PE  = 10
    _TIME_BASE_PE = 4
    _MLP_WIDTH    = 256
    _MLP_DEPTH    = 8

    # ------------------------------------------------------------------ setup

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

        # Standard 3DGS canonical parameters (initially empty)
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

        # Read arch params from args (set via --deform_mlp_width etc. in full_eval.py)
        pos_pe  = getattr(args, "deform_pos_pe",    self._POS_BASE_PE)
        time_pe = getattr(args, "deform_time_pe",   self._TIME_BASE_PE)
        W       = getattr(args, "deform_mlp_width", self._MLP_WIDTH)
        D       = getattr(args, "deform_mlp_depth", self._MLP_DEPTH)

        # Positional / time encoding buffers
        pos_freqs  = torch.FloatTensor([2 ** i for i in range(pos_pe)])
        time_freqs = torch.FloatTensor([2 ** i for i in range(time_pe)])

        pos_ch = 3 + 2 * 3 * pos_pe
        t_ch   = 1 + 2 * 1 * time_pe

        # MLP deformation network
        self._deform_net = _DeformMLP(pos_ch=pos_ch, t_ch=t_ch, W=W, D=D).cuda()
        self.register_buffer_pos_freqs  = pos_freqs.cuda()
        self.register_buffer_time_freqs = time_freqs.cuda()

        # Compatibility shim for scene/__init__.py
        self._deformation = _NoOpDeformation()

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
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color       = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        N = fused_point_cloud.shape[0]

        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        print(f"[DeformableMLP] Initialising {N} Gaussians")

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001
        )
        scales    = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots      = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
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
        Compute MLP-deformed Gaussian parameters at tau ∈ [-1, 1].

        Args:
            tau: normalised time in [-1, 1] (matches ODE model interface).

        Returns:
            xyz_t   : (N, 3) deformed positions
            scale_t : (N, 3) deformed log-scales
            rot_t   : (N, 4) deformed raw quaternions
        """
        t_raw = torch.tensor([(tau + 1.0) / 2.0], dtype=torch.float32, device="cuda")  # [0,1]

        # Encode position and time
        pos_emb  = poc_fre(self._xyz.detach(), self.register_buffer_pos_freqs)    # (N, 63)
        t_emb    = poc_fre(t_raw.unsqueeze(0), self.register_buffer_time_freqs)   # (1, 9)

        d_xyz, d_rot, d_scale = self._deform_net(pos_emb, t_emb)

        xyz_t   = self._xyz   + d_xyz
        rot_t   = self._rotation + d_rot
        scale_t = self._scaling  + d_scale
        return xyz_t, scale_t, rot_t

    def compute_ode_regulation(self) -> torch.Tensor:
        """No ODE regularisation for this baseline."""
        return torch.tensor(0.0, device="cuda", requires_grad=True)

    # ---------------------------------------------------------------- training setup

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        N = self.get_xyz.shape[0]

        self.xyz_gradient_accum  = torch.zeros((N, 1), device="cuda")
        self._deformation_accum  = torch.zeros((N, 3), device="cuda")
        self.denom               = torch.zeros((N, 1), device="cuda")

        deform_lr = getattr(training_args, "deform_lr", 2e-4)

        l = [
            {"params": [self._xyz],           "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc],   "lr": training_args.feature_lr,                               "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0,                        "name": "f_rest"},
            {"params": [self._opacity],       "lr": training_args.opacity_lr,                               "name": "opacity"},
            {"params": [self._scaling],       "lr": training_args.scaling_lr,                               "name": "scaling"},
            {"params": [self._rotation],      "lr": training_args.rotation_lr,                              "name": "rotation"},
            # Network params — all in one group; cat_tensors_to_optimizer skips multi-param groups
            {"params": list(self._deform_net.parameters()), "lr": deform_lr,   "name": "deform_net"},
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
            "xyz":     new_xyz,
            "f_dc":    new_features_dc,
            "f_rest":  new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
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
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )
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
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
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
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )
        new_xyz           = self._xyz[selected_pts_mask]
        new_features_dc   = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities     = self._opacity[selected_pts_mask]
        new_scaling       = self._scaling[selected_pts_mask]
        new_rotation      = self._rotation[selected_pts_mask]
        new_deform_table  = self._deformation_table[selected_pts_mask]
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation, new_deform_table)

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

    # ---------------------------------------------------------------- opacity reset

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
            if len(group["params"]) > 1:   # skip multi-param groups (e.g., deform_net)
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
            if len(group["params"]) > 1:   # skip multi-param groups
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
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
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
        self._xyz           = nn.Parameter(torch.tensor(xyz,           dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc   = nn.Parameter(torch.tensor(features_dc,  dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity       = nn.Parameter(torch.tensor(opacities,     dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling       = nn.Parameter(torch.tensor(scales,        dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation      = nn.Parameter(torch.tensor(rots,          dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def save_deformation(self, path):
        """Save MLP weights alongside the canonical PLY checkpoint."""
        torch.save(self._deform_net.state_dict(), os.path.join(path, "deform_net.pth"))
        torch.save(self._deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, "deformation_accum.pth"))

    def load_model(self, path):
        """Load MLP weights from a checkpoint directory."""
        net_path = os.path.join(path, "deform_net.pth")
        if not os.path.exists(net_path):
            print(f"[DeformableMLP] No deform_net.pth found at {path}")
            return
        self._deform_net.load_state_dict(torch.load(net_path, map_location="cuda"))
        self._deform_net = self._deform_net.cuda()
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
            self._deform_net.state_dict(),
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
            self.active_sh_degree,
            self._xyz,
            deform_state,
            self._deformation_table,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self._deform_net.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom              = denom
        self.optimizer.load_state_dict(opt_dict)
