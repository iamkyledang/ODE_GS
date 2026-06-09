"""
Neural ODE Velocity Field Baseline for Dynamic 3D Gaussian Splatting.

Based on:
  "EvoGS: Evolving Gaussian Splatting with Velocity Fields"
  Arnold Caleb, 2024.
  https://github.com/arnold-caleb/evogs

Architecture:
  - Positional encoding (Fourier features) for 3D position and time.
  - Shared velocity MLP mapping (pos_enc(x), time_enc(t)) → (dx/dt, dscale/dt, drot/dt).
  - Forward deformation integrates the velocity from t=0 to t_query using Euler steps.
  - Unlike displacement methods (deformable_MLP), velocities are predicted, not
    displacements.  The trajectory is the solution to an ODE, guaranteeing C¹
    continuity along time.

Key differences from deformable_MLP:
  - Predicts VELOCITY dx/dt = v(x, t) instead of displacement Δx = f(x, t).
  - Trajectories are computed by Euler integration with `neural_ode_steps` steps.
  - Ensures physically-coherent, smooth Gaussian trajectories.

Default hyperparameters (evogs dynerf/base settings):
  neural_ode_width   = 128   (MLP hidden width)
  neural_ode_depth   = 1     (MLP hidden layers)
  neural_ode_pos_pe  = 10    (positional encoding freq for xyz → 63-dim)
  neural_ode_time_pe = 4     (positional encoding freq for t  →  9-dim)
  neural_ode_steps   = 4     (Euler integration steps)

Usage in full_eval.py:
  Pass --model_class neural_ode to train.py / render.py.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
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
# Velocity MLP
# ─────────────────────────────────────────────────────────────────────────────

class _VelocityMLP(nn.Module):
    """
    Shared velocity field MLP.

    Input  : concatenation of pos_enc(xyz) and time_enc(t)
    Output : (v_xyz, v_quat, v_log_scale)  — instantaneous velocities

    Inspired by EvoGS velocity_field.VelocityField (arnold-caleb/evogs).
    Zero-initialised output heads to start near static (no motion).
    """

    def __init__(self, pos_ch: int, t_ch: int, W: int = 128, D: int = 1):
        super().__init__()
        in_ch = pos_ch + t_ch
        layers: list = [nn.Linear(in_ch, W), nn.ReLU()]
        for _ in range(D - 1):
            layers += [nn.Linear(W, W), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        # Output heads — small MLP per attribute velocity
        self.vel_xyz   = nn.Sequential(nn.ReLU(), nn.Linear(W, 3))
        self.vel_rot   = nn.Sequential(nn.ReLU(), nn.Linear(W, 4))
        self.vel_scale = nn.Sequential(nn.ReLU(), nn.Linear(W, 3))
        # Zero-initialise output layers → near-static at start of training
        for head in [self.vel_xyz, self.vel_rot, self.vel_scale]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x_enc: torch.Tensor, t_enc: torch.Tensor) -> tuple:
        # Broadcast t_enc if needed: t_enc may be (1, t_ch) or (N, t_ch)
        if t_enc.shape[0] == 1 and x_enc.shape[0] > 1:
            t_enc = t_enc.expand(x_enc.shape[0], -1)
        feat = torch.cat([x_enc, t_enc], dim=-1)
        h = self.trunk(feat)
        return self.vel_xyz(h), self.vel_rot(h), self.vel_scale(h)


# ─────────────────────────────────────────────────────────────────────────────
# Euler ODE integration
# ─────────────────────────────────────────────────────────────────────────────

def _euler_integrate(
    vel_net: _VelocityMLP,
    pos_freq_buf: torch.Tensor,
    t_freq_buf: torch.Tensor,
    xyz0: torch.Tensor,
    scale0: torch.Tensor,
    rot0: torch.Tensor,
    t_end: float,
    n_steps: int = 4,
) -> tuple:
    """
    Euler integration of velocity field from t=0 to t_end.

    State (xyz, scale, rot) is updated with constant step dt = t_end / n_steps.
    The velocity is re-evaluated at each step position (1st-order accurate).
    """
    if abs(t_end) < 1e-7:
        return xyz0, scale0, rot0

    dt = t_end / n_steps
    x = xyz0
    s = scale0
    r = rot0

    for i in range(n_steps):
        t_cur = i * dt
        t_buf = torch.full((1, 1), t_cur, dtype=torch.float32, device=x.device)
        x_enc = poc_fre(x, pos_freq_buf)                          # (N, pos_ch)
        t_enc = poc_fre(t_buf, t_freq_buf)                        # (1, t_ch)
        vx, vr, vs = vel_net(x_enc, t_enc)
        x = x + dt * vx
        # Clamp log-scales: prevents exp() overflow → NaN in the rasteriser.
        s = torch.clamp(s + dt * vs, -10.0, 10.0)
        # Normalise after each step: F.normalize(zero-vector) = NaN, so keeping
        # the quaternion near unit-norm is critical for fine-training stability.
        r = torch.nn.functional.normalize(r + dt * vr, dim=-1)

    return x, s, r


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility shims for scene/__init__.py
# ─────────────────────────────────────────────────────────────────────────────

class _NoOpDeformNet:
    """Dummy deformation net — velocity field has no AABB constraint."""
    def set_aabb(self, xyz_max, xyz_min):
        pass


class _NoOpDeformation:
    def __init__(self):
        self.deformation_net = _NoOpDeformNet()

    def set_aabb(self, xyz_max, xyz_min):
        self.deformation_net.set_aabb(xyz_max, xyz_min)


# ─────────────────────────────────────────────────────────────────────────────
# GaussianModel — Neural ODE velocity field baseline
# ─────────────────────────────────────────────────────────────────────────────

class GaussianModel:
    """
    Dynamic 3D Gaussian Model with Neural ODE velocity field.

    Canonical state: standard 3DGS parameters (_xyz, _scaling, _rotation, …)
    Deformation    : shared velocity MLP v(x,t) integrated via Euler ODE.

    Key difference from deformable_MLP (displacement baseline):
      deformable_MLP  → learns Δx = f(x, t)   (independent per-frame offsets)
      neural_ode      → learns dx/dt = v(x, t) then integrates x(t) from t=0

    Architecture defaults match evogs dynerf/base settings:
      width=128, depth=1, pos_pe=10, time_pe=4, euler_steps=4.
    """

    # Default architecture hyperparameters (evogs dynerf/base)
    _VEL_WIDTH    = 128
    _VEL_DEPTH    = 1
    _POS_BASE_PE  = 10
    _TIME_BASE_PE = 4
    _ODE_STEPS    = 4

    # ── setup ────────────────────────────────────────────────────────────────

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return strip_symmetric(actual_covariance)

        self.scaling_activation         = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation      = build_covariance_from_scaling_rotation
        self.opacity_activation         = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation        = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, args):
        self.active_sh_degree = 0
        self.max_sh_degree    = sh_degree

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

        # Compatibility shim for scene/__init__.py set_aabb calls
        self._deformation = _NoOpDeformation()

        # Read arch params from args
        pos_pe    = int(getattr(args, "neural_ode_pos_pe",  self._POS_BASE_PE))
        time_pe   = int(getattr(args, "neural_ode_time_pe", self._TIME_BASE_PE))
        W         = int(getattr(args, "neural_ode_width",   self._VEL_WIDTH))
        D         = int(getattr(args, "neural_ode_depth",   self._VEL_DEPTH))
        self._ode_steps = int(getattr(args, "neural_ode_steps", self._ODE_STEPS))

        # Positional encoding buffers (stored as plain CUDA tensors, not nn.Parameter)
        pos_freqs  = torch.from_numpy(
            np.array([2 ** i for i in range(pos_pe)], dtype=np.float32)
        ).cuda()
        time_freqs = torch.from_numpy(
            np.array([2 ** i for i in range(time_pe)], dtype=np.float32)
        ).cuda()

        self.register_buffer_pos_freqs  = pos_freqs
        self.register_buffer_time_freqs = time_freqs

        pos_ch = 3 + 2 * 3 * pos_pe   # 3 + 2*3*10 = 63
        t_ch   = 1 + 2 * 1 * time_pe  # 1 + 2*1*4  = 9

        self._vel_net = _VelocityMLP(pos_ch=pos_ch, t_ch=t_ch, W=W, D=D).cuda()

        self.setup_functions()

    # ── 3DGS standard properties ─────────────────────────────────────────────

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

    # ── deformation (ODE interface used by render_ode) ───────────────────────

    def apply_ode_deformation(self, tau: float) -> tuple:
        """
        Integrate velocity field from t=0 to t=(tau+1)/2 using Euler steps.

        Args:
            tau : normalised time in [-1, 1]  (render_ode passes tau = 2*t - 1)

        Returns:
            xyz_t   : (N, 3) deformed positions
            scale_t : (N, 3) deformed log-scales  (raw, before exp activation)
            rot_t   : (N, 4) deformed raw quaternions
        """
        t_val = (tau + 1.0) / 2.0   # map [-1, 1] → [0, 1]

        xyz_t, scale_t, rot_t = _euler_integrate(
            vel_net=self._vel_net,
            pos_freq_buf=self.register_buffer_pos_freqs,
            t_freq_buf=self.register_buffer_time_freqs,
            xyz0=self._xyz,
            scale0=self._scaling,
            # Use normalised rotation as the integration start so the canonical
            # quaternion is always a proper unit quaternion even if the raw
            # parameter drifts slightly away from the unit sphere during Adam updates.
            rot0=torch.nn.functional.normalize(self._rotation, dim=-1),
            t_end=t_val,
            n_steps=self._ode_steps,
        )
        return xyz_t, scale_t, rot_t

    def compute_ode_regulation(self) -> torch.Tensor:
        """
        Velocity coherence regularisation.

        Penalises large velocities at randomly-sampled Gaussian positions and
        times.  Encourages the velocity field to remain near-zero when not
        needed, preventing spurious motion artefacts.
        """
        if self._xyz.shape[0] == 0:
            return torch.tensor(0.0, device="cuda", requires_grad=True)

        # Sample up to 1024 canonical positions
        N = min(1024, self._xyz.shape[0])
        idx = torch.randperm(self._xyz.shape[0], device="cuda")[:N]
        pts = self._xyz[idx].detach()

        reg = torch.tensor(0.0, device="cuda")
        for _ in range(2):
            t_val = torch.rand(1).item()
            t_buf = torch.full((1, 1), t_val, dtype=torch.float32, device="cuda")
            x_enc = poc_fre(pts, self.register_buffer_pos_freqs)
            t_enc = poc_fre(t_buf, self.register_buffer_time_freqs).expand(N, -1)
            vx, vr, vs = self._vel_net(x_enc, t_enc)
            reg = reg + vx.pow(2).mean() + vr.pow(2).mean() + vs.pow(2).mean()

        return reg / 2.0

    # ── initialisation ────────────────────────────────────────────────────────

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, maxtime: int = 0):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color       = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros(
            (fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)
        ).float().cuda()
        features[:, :3, 0] = fused_color

        N = fused_point_cloud.shape[0]
        print(f"[NeuralODE] Initialising {N} Gaussians  "
              f"(vel_net W={self._vel_net.trunk[0].out_features}, steps={self._ode_steps})")

        dist2      = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7)
        scales     = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots       = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        opacities  = inverse_sigmoid(0.1 * torch.ones((N, 1), dtype=torch.float, device="cuda"))

        self._xyz            = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc    = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest  = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling        = nn.Parameter(scales.requires_grad_(True))
        self._rotation       = nn.Parameter(rots.requires_grad_(True))
        self._opacity        = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D     = torch.zeros((N,), device="cuda")
        self._deformation_table = torch.gt(torch.ones((N,), device="cuda"), 0)

    # ── training setup ────────────────────────────────────────────────────────

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        N = self.get_xyz.shape[0]

        self.xyz_gradient_accum  = torch.zeros((N, 1), device="cuda")
        self._deformation_accum  = torch.zeros((N, 3), device="cuda")
        self.denom               = torch.zeros((N, 1), device="cuda")

        vel_lr_init  = getattr(training_args, "ode_lr_init",       1e-4)
        vel_lr_final = getattr(training_args, "ode_lr_final",       1e-5)
        vel_lr_delay = getattr(training_args, "ode_lr_delay_mult",  0.01)

        l = [
            {"params": [self._xyz],           "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc],   "lr": training_args.feature_lr,                               "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0,                        "name": "f_rest"},
            {"params": [self._opacity],       "lr": training_args.opacity_lr,                               "name": "opacity"},
            {"params": [self._scaling],       "lr": training_args.scaling_lr,                               "name": "scaling"},
            {"params": [self._rotation],      "lr": training_args.rotation_lr,                              "name": "rotation"},
            {"params": list(self._vel_net.parameters()), "lr": vel_lr_init,                                 "name": "vel_net"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.vel_scheduler_args = get_expon_lr_func(
            lr_init=vel_lr_init,
            lr_final=vel_lr_final,
            lr_delay_mult=vel_lr_delay,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self.xyz_scheduler_args(iteration)
            elif param_group["name"] == "vel_net":
                param_group["lr"] = self.vel_scheduler_args(iteration)

    # ── densification / pruning helpers ───────────────────────────────────────

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor_grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:   # skip multi-param groups (vel_net)
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"]    = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_mask)
        self._xyz            = optimizable_tensors["xyz"]
        self._features_dc    = optimizable_tensors["f_dc"]
        self._features_rest  = optimizable_tensors["f_rest"]
        self._opacity        = optimizable_tensors["opacity"]
        self._scaling        = optimizable_tensors["scaling"]
        self._rotation       = optimizable_tensors["rotation"]
        new_N = self.get_xyz.shape[0]
        self.xyz_gradient_accum  = torch.zeros((new_N, 1), device="cuda")
        self._deformation_accum  = torch.zeros((new_N, 3), device="cuda")
        self.denom               = torch.zeros((new_N, 1), device="cuda")
        self.max_radii2D         = self.max_radii2D[valid_mask]
        self._deformation_table  = self._deformation_table[valid_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:   # skip vel_net
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                               new_opacities, new_scaling, new_rotation):
        d = {
            "xyz":     new_xyz,
            "f_dc":    new_features_dc,
            "f_rest":  new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation":new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz            = optimizable_tensors["xyz"]
        self._features_dc    = optimizable_tensors["f_dc"]
        self._features_rest  = optimizable_tensors["f_rest"]
        self._opacity        = optimizable_tensors["opacity"]
        self._scaling        = optimizable_tensors["scaling"]
        self._rotation       = optimizable_tensors["rotation"]
        num_new = new_xyz.shape[0]
        new_N = self.get_xyz.shape[0]
        self.xyz_gradient_accum  = torch.zeros((new_N, 1), device="cuda")
        self._deformation_accum  = torch.zeros((new_N, 3), device="cuda")
        self.denom               = torch.zeros((new_N, 1), device="cuda")
        self.max_radii2D         = torch.zeros((new_N,),   device="cuda")
        self._deformation_table  = torch.cat([
            self._deformation_table,
            torch.ones(num_new, dtype=torch.bool, device="cuda"),
        ], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_pts  = self._xyz.shape[0]
        padded_grad = torch.zeros(n_init_pts, device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.logical_and(
            padded_grad >= grad_threshold,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )
        stds    = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means   = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots    = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                   + self._xyz[selected_pts_mask].repeat(N, 1))
        new_scaling  = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation      = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc   = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacities     = self._opacity[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacities, new_scaling, new_rotation)
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), dtype=torch.bool, device="cuda"),
        ))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.logical_and(
            grads.squeeze() >= grad_threshold,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )
        new_xyz           = self._xyz[selected_pts_mask]
        new_features_dc   = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities     = self._opacity[selected_pts_mask]
        new_scaling       = self._scaling[selected_pts_mask]
        new_rotation      = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacities, new_scaling, new_rotation)

    def densify(self, max_grad, opacity_threshold, scene_extent, size_threshold,
                clone_extent=5, split_extent=5, model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, scene_extent)
        self.densify_and_split(grads, max_grad, scene_extent)

    def prune(self, max_grad, opacity_threshold, scene_extent, size_threshold):
        prune_mask = (self.get_opacity < opacity_threshold).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def grow(self, *args, **kwargs):
        pass

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # ── optimizer tensor utilities ────────────────────────────────────────────

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

    # ── PLY save / load ───────────────────────────────────────────────────────

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
        self._xyz           = nn.Parameter(torch.from_numpy(xyz.astype(np.float32)).cuda().requires_grad_(True))
        self._features_dc   = nn.Parameter(torch.from_numpy(features_dc.astype(np.float32)).cuda().transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.from_numpy(features_extra.astype(np.float32)).cuda().transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity       = nn.Parameter(torch.from_numpy(opacities.astype(np.float32)).cuda().requires_grad_(True))
        self._scaling       = nn.Parameter(torch.from_numpy(scales.astype(np.float32)).cuda().requires_grad_(True))
        self._rotation      = nn.Parameter(torch.from_numpy(rots.astype(np.float32)).cuda().requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def save_deformation(self, path):
        """Save velocity network weights alongside the canonical PLY checkpoint."""
        torch.save(self._vel_net.state_dict(), os.path.join(path, "vel_net.pth"))
        torch.save(self._deformation_table,    os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,    os.path.join(path, "deformation_accum.pth"))

    def load_model(self, path):
        """Load velocity network weights from a checkpoint directory."""
        net_path = os.path.join(path, "vel_net.pth")
        if not os.path.exists(net_path):
            print(f"[NeuralODE] No vel_net.pth found at {path}")
            return
        self._vel_net.load_state_dict(torch.load(net_path, map_location="cuda"))
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(
                os.path.join(path, "deformation_table.pth"), map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(
                os.path.join(path, "deformation_accum.pth"), map_location="cuda")
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device="cuda")

    # ── checkpoint capture / restore ─────────────────────────────────────────

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._vel_net.state_dict(),
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
            vel_state,
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
        self._vel_net.load_state_dict(vel_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom              = denom
        self.optimizer.load_state_dict(opt_dict)
