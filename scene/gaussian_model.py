"""
GaussianModel: Dynamic 3D Gaussian Splatting with explicit 3D ODE-based
temporal evolution.

Implements the model described in:
  "Dynamic 3D Gaussian Splatting with Explicit Real-Valued ODE-Based
   Mean and Covariance Evolution"

Each Gaussian has per-primitive ODE parameters:
  Mean evolution (3D real-valued ODE):
    dx_i/dtau = A_i * x_i + b_i,   A_i in R^{3x3}, b_i in R^3, x_i(0) = x_i^0
    mu_i(tau) = mu_i^0 + x_i(tau)

  Covariance scale evolution:
    ds_{ij}/dtau = kappa_{ij}  =>  s_{ij}(tau) = s_{ij}^0 + kappa_{ij} * tau

  Covariance orientation evolution (SO(3)):
    dR_i/dtau = omega_hat_i * R_i,  omega_i in R^3
    R_i(tau)  = exp(omega_hat_i * tau) * R_i^0

No neural deformation field is used. The rendering pipeline is the original
3DGS rasterizer — no changes to the renderer.

Training stages (mirrors the original 4DGaussians setup):
  coarse stage: renders canonical Gaussians (tau=0, no ODE displacement)
  fine   stage: renders ODE-deformed Gaussians at the camera timestamp
"""

import torch
import numpy as np
from torch import nn
import os

from plyfile import PlyData, PlyElement
from random import randint

from simple_knn._C import distCUDA2
from utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func, build_rotation,
    strip_symmetric, build_scaling_rotation,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

from scene.deformation import (
    solve_3d_ode,
    evolve_covariance,
    quaternion_to_rotation_matrix,
)


# ---------------------------------------------------------------------------
# Compatibility shim so that scene/__init__.py set_aabb call does not crash.
# The ODE model has no spatial grid so AABB is ignored.
# ---------------------------------------------------------------------------

class _NoOpDeformNet:
    def set_aabb(self, xyz_max, xyz_min):
        pass


class _NoOpDeformation:
    def __init__(self):
        self.deformation_net = _NoOpDeformNet()

    def set_aabb(self, xyz_max, xyz_min):
        pass


# ---------------------------------------------------------------------------
# ODE parameter names (for densification / pruning bookkeeping)
# ---------------------------------------------------------------------------

_ODE_PARAM_NAMES = [
    "ode_A_flat",    # (N, 9)   row-major 3x3 dynamics matrix
    "ode_b",         # (N, 3)   drift / forcing vector
    "ode_x0",        # (N, 3)   initial 3D displacement at tau=0
    "ode_kappa",     # (N, 3)   log-scale rate
    "ode_omega_cov", # (N, 3)   angular velocity for covariance rotation (SO(3))
]


def _zero_ode_tensors(n: int, device: str = "cuda") -> dict:
    """
    Return zero-initialised ODE parameter tensors for n new Gaussians.
    Used when adding Gaussians during densification.
    """
    return {
        "ode_A_flat":    torch.zeros(n, 9, device=device),
        "ode_b":         torch.zeros(n, 3, device=device),
        "ode_x0":        torch.zeros(n, 3, device=device),
        "ode_kappa":     torch.zeros(n, 3, device=device),
        "ode_omega_cov": torch.zeros(n, 3, device=device),
    }


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class GaussianModel:
    """
    Dynamic 3D Gaussian Model with per-Gaussian ODE-based temporal parameterisation.

    Canonical state (at tau = 0):
        _xyz            : (N, 3)   Gaussian centres
        _scaling        : (N, 3)   log-scales (exp gives actual scale)
        _rotation       : (N, 4)   unit quaternions [w, x, y, z]
        _opacity        : (N, 1)   logit-opacities (sigmoid gives actual opacity)
        _features_dc    : (N, 1, 3) DC spherical harmonic coefficients
        _features_rest  : (N, K, 3) higher-order SH coefficients

    Per-Gaussian ODE parameters (all trainable):
        _ode_A_flat    : (N, 9)   row-major 3x3 dynamics matrix A
        _ode_b         : (N, 3)   drift vector b
        _ode_x0        : (N, 3)   initial 3D displacement x(0)
        _ode_kappa     : (N, 3)   log-scale rate
        _ode_omega_cov : (N, 3)   angular velocity for covariance rotation
    """

    # ------------------------------------------------------------------
    # Setup / init
    # ------------------------------------------------------------------

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

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

        # ODE mean parameters: 3D real-valued ODE
        self._ode_A_flat    = torch.empty(0)   # (N, 9)
        self._ode_b         = torch.empty(0)   # (N, 3)
        self._ode_x0        = torch.empty(0)   # (N, 3)

        # ODE covariance parameters
        self._ode_kappa     = torch.empty(0)   # (N, 3)  log-scale rate
        self._ode_omega_cov = torch.empty(0)   # (N, 3)  angular velocity

        # Compatibility shim: scene/__init__.py calls _deformation.deformation_net.set_aabb()
        self._deformation = _NoOpDeformation()

        self.setup_functions()

    # ------------------------------------------------------------------
    # Properties (mirror GaussianModel interface)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Initialisation from point cloud
    # ------------------------------------------------------------------

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color       = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        N = fused_point_cloud.shape[0]

        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        print(f"[GaussianModel] Initialising {N} Gaussians")

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001
        )
        scales    = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots      = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1   # identity quaternion  w=1
        opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), dtype=torch.float, device="cuda"))

        # Canonical 3DGS parameters
        self._xyz           = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc   = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling       = nn.Parameter(scales.requires_grad_(True))
        self._rotation      = nn.Parameter(rots.requires_grad_(True))
        self._opacity       = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D    = torch.zeros(N, device="cuda")
        self._deformation_table = torch.gt(torch.ones(N, device="cuda"), 0)

        # ODE parameters — zero-initialised (no motion at start of training)
        self._init_ode_params(N)

    def _init_ode_params(self, N: int):
        """Zero-initialise all ODE parameters for N Gaussians."""
        device = "cuda"
        self._ode_A_flat    = nn.Parameter(torch.zeros(N, 9, device=device))
        self._ode_b         = nn.Parameter(torch.zeros(N, 3, device=device))
        self._ode_x0        = nn.Parameter(torch.zeros(N, 3, device=device))
        self._ode_kappa     = nn.Parameter(torch.zeros(N, 3, device=device))
        self._ode_omega_cov = nn.Parameter(torch.zeros(N, 3, device=device))

    # ------------------------------------------------------------------
    # Core: compute ODE-deformed Gaussian parameters at time tau
    # ------------------------------------------------------------------

    def apply_ode_deformation(self, tau: float):
        """
        Compute deformed Gaussian parameters at normalised time tau in [-1, 1].

        The dataset timestamp t in [0, 1] is converted to tau in [-1, 1] by
        the renderer: tau = 2*t - 1.

        For each Gaussian i at time tau:
          Mean:
            x_i(tau) = solve_3d_ode(A_i, b_i, x0_i, tau)
            mu_i(tau) = mu_i^0 + x_i(tau)

          Log-scale:
            s_j(tau) = s_j^0 + kappa_j * tau

          Quaternion:
            q(tau) = exp(omega_hat * tau) * q^0   (SO(3) rotation)

        Args:
            tau: float, normalised time in [-1, 1].

        Returns:
            xyz_t   : (N, 3) deformed Gaussian centres.
            scale_t : (N, 3) deformed log-scales (raw, before exp activation).
            rot_t   : (N, 4) deformed unit quaternions.
        """
        # Mean evolution: mu_i(tau) = mu_i^0 + x_i(tau)
        x_t   = solve_3d_ode(self._ode_A_flat, self._ode_b, self._ode_x0, tau)
        xyz_t = self._xyz + x_t

        # Covariance evolution
        scale_t, rot_t = evolve_covariance(
            self._scaling, self._rotation,
            self._ode_kappa, self._ode_omega_cov,
            tau,
        )

        return xyz_t, scale_t, rot_t

    # ------------------------------------------------------------------
    # ODE regularisation loss
    # ------------------------------------------------------------------

    def compute_ode_regulation(self) -> torch.Tensor:
        """
        ODE regularisation loss (evaluated at tau = 0):

          L_ode = L_traj + L_omega + L_s

          L_traj  = mean_i ||A_i * x0_i + b_i||^2   (velocity at tau=0)
          L_omega = mean_i ||omega_cov_i||^2
          L_s     = mean_i ||kappa_i||^2
        """
        A   = self._ode_A_flat.reshape(-1, 3, 3)
        dx0 = torch.bmm(A, self._ode_x0.unsqueeze(-1)).squeeze(-1) + self._ode_b
        L_traj  = (dx0 ** 2).mean()
        L_omega = (self._ode_omega_cov ** 2).mean()
        L_s     = (self._ode_kappa ** 2).mean()
        return L_traj + L_omega + L_s

    # ------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        N = self.get_xyz.shape[0]

        self.xyz_gradient_accum  = torch.zeros((N, 1), device="cuda")
        self._deformation_accum  = torch.zeros((N, 3), device="cuda")
        self.denom               = torch.zeros((N, 1), device="cuda")

        ode_lr_init  = getattr(training_args, "ode_lr_init",  1e-4) * self.spatial_lr_scale
        ode_lr_final = getattr(training_args, "ode_lr_final", 1e-5) * self.spatial_lr_scale
        ode_lr_delay = getattr(training_args, "ode_lr_delay_mult", 0.01)

        l = [
            {"params": [self._xyz],           "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc],   "lr": training_args.feature_lr,                               "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0,                        "name": "f_rest"},
            {"params": [self._opacity],       "lr": training_args.opacity_lr,                               "name": "opacity"},
            {"params": [self._scaling],       "lr": training_args.scaling_lr,                               "name": "scaling"},
            {"params": [self._rotation],      "lr": training_args.rotation_lr,                              "name": "rotation"},
            # 3D ODE mean parameters
            {"params": [self._ode_A_flat],    "lr": ode_lr_init, "name": "ode_A_flat"},
            {"params": [self._ode_b],         "lr": ode_lr_init, "name": "ode_b"},
            {"params": [self._ode_x0],        "lr": ode_lr_init, "name": "ode_x0"},
            # Covariance ODE parameters
            {"params": [self._ode_kappa],     "lr": ode_lr_init, "name": "ode_kappa"},
            {"params": [self._ode_omega_cov], "lr": ode_lr_init, "name": "ode_omega_cov"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.ode_scheduler_args = get_expon_lr_func(
            lr_init=ode_lr_init,
            lr_final=ode_lr_final,
            lr_delay_mult=ode_lr_delay,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self.xyz_scheduler_args(iteration)
            elif param_group["name"].startswith("ode_"):
                param_group["lr"] = self.ode_scheduler_args(iteration)

    # ------------------------------------------------------------------
    # Densification helpers
    # ------------------------------------------------------------------

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation, new_deformation_table):
        """
        Extend ALL parameter tensors when new Gaussians are added.
        New primitives are zero-initialised for all ODE params.
        """
        n_new = new_xyz.shape[0]
        d = {
            "xyz":      new_xyz,
            "f_dc":     new_features_dc,
            "f_rest":   new_features_rest,
            "opacity":  new_opacities,
            "scaling":  new_scaling,
            "rotation": new_rotation,
        }
        d.update(_zero_ode_tensors(n_new))
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz           = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]
        self._ode_A_flat    = optimizable_tensors["ode_A_flat"]
        self._ode_b         = optimizable_tensors["ode_b"]
        self._ode_x0        = optimizable_tensors["ode_x0"]
        self._ode_kappa     = optimizable_tensors["ode_kappa"]
        self._ode_omega_cov = optimizable_tensors["ode_omega_cov"]

        self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], -1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom              = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D        = torch.zeros(self.get_xyz.shape[0],      device="cuda")

    def prune_points(self, mask):
        """Remove Gaussians where mask is True."""
        valid = ~mask
        optimizable_tensors = self._prune_optimizer(valid)

        self._xyz           = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]
        self._ode_A_flat    = optimizable_tensors["ode_A_flat"]
        self._ode_b         = optimizable_tensors["ode_b"]
        self._ode_x0        = optimizable_tensors["ode_x0"]
        self._ode_kappa     = optimizable_tensors["ode_kappa"]
        self._ode_omega_cov = optimizable_tensors["ode_omega_cov"]

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

        new_xyz         = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)                         + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling     = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation    = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity     = self._opacity[selected_pts_mask].repeat(N, 1)
        new_deform_table = self._deformation_table[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation, new_deform_table,
        )
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
        ))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent,
                          density_threshold=20, displacement_scale=20,
                          model_path=None, iteration=None, stage=None):
        grads_accum_mask  = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            grads_accum_mask,
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
            new_opacities, new_scaling, new_rotation, new_deform_table,
        )

    def densify(self, max_grad, min_opacity, extent, max_screen_size,
                density_threshold, displacement_scale, model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(
            grads, max_grad, extent, density_threshold, displacement_scale,
            model_path, iteration, stage,
        )
        self.densify_and_split(grads, max_grad, extent)

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    # ------------------------------------------------------------------
    # Opacity reset
    # ------------------------------------------------------------------

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # ------------------------------------------------------------------
    # Optimizer utilities
    # ------------------------------------------------------------------

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"]    = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
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
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
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
            key=lambda x: int(x.split("_")[-1]),
        )
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz           = nn.Parameter(torch.tensor(xyz,            dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc   = nn.Parameter(torch.tensor(features_dc,   dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity       = nn.Parameter(torch.tensor(opacities,      dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling       = nn.Parameter(torch.tensor(scales,         dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation      = nn.Parameter(torch.tensor(rots,           dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def save_deformation(self, path):
        """Save ODE parameters alongside the canonical PLY checkpoint."""
        ode_state = {
            "ode_A_flat":    self._ode_A_flat.detach().cpu(),
            "ode_b":         self._ode_b.detach().cpu(),
            "ode_x0":        self._ode_x0.detach().cpu(),
            "ode_kappa":     self._ode_kappa.detach().cpu(),
            "ode_omega_cov": self._ode_omega_cov.detach().cpu(),
        }
        torch.save(ode_state, os.path.join(path, "ode_deformation.pth"))
        torch.save(self._deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, "deformation_accum.pth"))

    def load_model(self, path):
        """Load ODE parameters from a checkpoint directory."""
        ode_path = os.path.join(path, "ode_deformation.pth")
        if not os.path.exists(ode_path):
            print(f"[GaussianModel] No ode_deformation.pth found at {path}")
            return

        ode_state = torch.load(ode_path, map_location="cuda")
        self._ode_A_flat    = nn.Parameter(ode_state["ode_A_flat"].cuda())
        self._ode_b         = nn.Parameter(ode_state.get("ode_b", ode_state.get("ode_b_vec")).cuda())
        self._ode_x0        = nn.Parameter(ode_state["ode_x0"].cuda())
        self._ode_kappa     = nn.Parameter(ode_state["ode_kappa"].cuda())
        self._ode_omega_cov = nn.Parameter(ode_state.get("ode_omega_cov", ode_state.get("ode_omega_vec")).cuda())

        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(
                os.path.join(path, "deformation_table.pth"), map_location="cuda"
            )
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(
                os.path.join(path, "deformation_accum.pth"), map_location="cuda"
            )
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device="cuda")

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            {
                "ode_A_flat":    self._ode_A_flat,
                "ode_b":         self._ode_b,
                "ode_x0":        self._ode_x0,
                "ode_kappa":     self._ode_kappa,
                "ode_omega_cov": self._ode_omega_cov,
            },
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
            ode_state,
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

        self._ode_A_flat    = ode_state["ode_A_flat"]
        self._ode_b         = ode_state["ode_b"]
        self._ode_x0        = ode_state["ode_x0"]
        self._ode_kappa     = ode_state["ode_kappa"]
        self._ode_omega_cov = ode_state["ode_omega_cov"]

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom              = denom
        self.optimizer.load_state_dict(opt_dict)

    # ------------------------------------------------------------------
    # Trajectory / motion analysis utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_trajectory(self, tau_list):
        """
        Compute Gaussian centre trajectories over a list of time steps.

        Args:
            tau_list: list of T tau values in [-1, 1].
        Returns:
            trajectories: (T, N, 3) Gaussian positions at each tau.
        """
        positions = []
        for tau in tau_list:
            xyz_t, _, _ = self.apply_ode_deformation(float(tau))
            positions.append(xyz_t.detach().cpu())
        return torch.stack(positions, dim=0)   # (T, N, 3)

    @torch.no_grad()
    def get_velocity_and_acceleration(self, tau_list):
        """
        Estimate per-Gaussian velocity and acceleration from the ODE trajectory.

        Args:
            tau_list: list of T equally-spaced tau values.
        Returns:
            velocities:    (T-1, N, 3) finite-difference velocity.
            accelerations: (T-2, N, 3) finite-difference acceleration.
        """
        traj = self.get_trajectory(tau_list)          # (T, N, 3)
        dt   = tau_list[1] - tau_list[0]
        velocities    = (traj[1:] - traj[:-1]) / dt   # (T-1, N, 3)
        accelerations = (velocities[1:] - velocities[:-1]) / dt  # (T-2, N, 3)
        return velocities, accelerations
