"""
Polynomial Approximation Baseline for Dynamic 3D Gaussian Splatting.

Architecture:
  - Per-Gaussian polynomial coefficients representing temporal trajectories.
  - Each Gaussian stores D polynomial coefficient vectors for each attribute:
      delta_attr(tau) = Σ_{d=1}^{D} coeff_d * tau^d
    where tau ∈ [-1, 1] is normalised time.
  - No constant term (d=0) so the canonical state is preserved at tau=0.
  - D=4 polynomial degree by default.
  - Attributes: position (3), rotation delta (4), log-scale delta (3).
  - No neural network — purely parametric, interpretable representation.

Usage in full_eval.py:
  Pass --model_type polynomial_approx to train.py / render.py.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

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
# Compatibility shim
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
# GaussianModel — Polynomial approximation baseline
# ─────────────────────────────────────────────────────────────────────────────

class GaussianModel:
    """
    Dynamic 3D Gaussian Model with per-Gaussian polynomial temporal parameterisation.

    Each Gaussian stores D coefficient tensors per attribute:
        _poly_pos   : (N, D, 3)  — coefficients for polynomial Δxyz
        _poly_rot   : (N, D, 4)  — coefficients for polynomial Δquat
        _poly_scale : (N, D, 3)  — coefficients for polynomial Δlog_scale

    At time tau ∈ [-1, 1]:
        Δattr(tau) = Σ_{d=1}^{D} coeff[:, d-1, :] * tau^d

    tau=0 corresponds to the canonical (reference) frame (Δattr=0 exactly).
    """

    # Polynomial degree: terms tau^1 through tau^D
    D = 4

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

        # Architecture param — overridable via --poly_D in full_eval.py
        self.D = int(getattr(args, "poly_D", self.D))

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

        # Per-Gaussian polynomial parameters (initialised in create_from_pcd)
        self._poly_pos   = torch.empty(0)
        self._poly_rot   = torch.empty(0)
        self._poly_scale = torch.empty(0)

        # Compatibility shim
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
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points).astype(np.float32)).cuda()
        fused_color       = RGB2SH(torch.from_numpy(np.asarray(pcd.colors).astype(np.float32)).cuda())
        N = fused_point_cloud.shape[0]

        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        print(f"[PolyApprox] Initialising {N} Gaussians (degree D={self.D})")

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

        # Zero-init polynomial parameters
        self._poly_pos   = nn.Parameter(torch.zeros(N, self.D, 3, device="cuda"))
        self._poly_rot   = nn.Parameter(torch.zeros(N, self.D, 4, device="cuda"))
        self._poly_scale = nn.Parameter(torch.zeros(N, self.D, 3, device="cuda"))

    # ---------------------------------------------------------------- deformation

    def _poly_powers(self, tau: float) -> torch.Tensor:
        """
        Compute [tau^1, tau^2, ..., tau^D] as a (1, D, 1) tensor.
        """
        return torch.from_numpy(
            np.array([tau ** d for d in range(1, self.D + 1)], dtype=np.float32)
        ).cuda().view(1, self.D, 1)

    def apply_ode_deformation(self, tau: float):
        """
        Compute polynomial-deformed Gaussian parameters at tau ∈ [-1, 1].

        Δattr(tau) = Σ_{d=1}^{D} coeff[:, d-1, :] * tau^d

        Returns:
            xyz_t   : (N, 3)
            scale_t : (N, 3)
            rot_t   : (N, 4)
        """
        powers = self._poly_powers(tau)  # (1, D, 1)

        d_xyz   = (self._poly_pos   * powers).sum(dim=1)  # (N, 3)
        d_rot   = (self._poly_rot   * powers).sum(dim=1)  # (N, 4)
        d_scale = (self._poly_scale * powers).sum(dim=1)  # (N, 3)

        return self._xyz + d_xyz, self._scaling + d_scale, self._rotation + d_rot

    def compute_ode_regulation(self) -> torch.Tensor:
        """L2 regularisation on polynomial coefficients to encourage smoothness."""
        reg = (self._poly_pos   ** 2).mean() + \
              (self._poly_rot   ** 2).mean() + \
              (self._poly_scale ** 2).mean()
        return reg

    # ---------------------------------------------------------------- training setup

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        N = self.get_xyz.shape[0]

        self.xyz_gradient_accum  = torch.zeros((N, 1), device="cuda")
        self._deformation_accum  = torch.zeros((N, 3), device="cuda")
        self.denom               = torch.zeros((N, 1), device="cuda")

        poly_lr = getattr(training_args, "poly_lr",
                          getattr(training_args, "ode_lr_init", 1e-4))

        l = [
            {"params": [self._xyz],          "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc],  "lr": training_args.feature_lr,                               "name": "f_dc"},
            {"params": [self._features_rest],"lr": training_args.feature_lr / 20.0,                        "name": "f_rest"},
            {"params": [self._opacity],      "lr": training_args.opacity_lr,                               "name": "opacity"},
            {"params": [self._scaling],      "lr": training_args.scaling_lr,                               "name": "scaling"},
            {"params": [self._rotation],     "lr": training_args.rotation_lr,                              "name": "rotation"},
            {"params": [self._poly_pos],     "lr": poly_lr, "name": "poly_pos"},
            {"params": [self._poly_rot],     "lr": poly_lr, "name": "poly_rot"},
            {"params": [self._poly_scale],   "lr": poly_lr, "name": "poly_scale"},
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
        n_new = new_xyz.shape[0]
        d = {
            "xyz":        new_xyz,
            "f_dc":       new_features_dc,
            "f_rest":     new_features_rest,
            "opacity":    new_opacities,
            "scaling":    new_scaling,
            "rotation":   new_rotation,
            "poly_pos":   torch.zeros(n_new, self.D, 3, device="cuda"),
            "poly_rot":   torch.zeros(n_new, self.D, 4, device="cuda"),
            "poly_scale": torch.zeros(n_new, self.D, 3, device="cuda"),
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz           = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]
        self._poly_pos      = optimizable_tensors["poly_pos"]
        self._poly_rot      = optimizable_tensors["poly_rot"]
        self._poly_scale    = optimizable_tensors["poly_scale"]

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
        self._poly_pos      = optimizable_tensors["poly_pos"]
        self._poly_rot      = optimizable_tensors["poly_rot"]
        self._poly_scale    = optimizable_tensors["poly_scale"]
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
            "poly_pos":   self._poly_pos.detach().cpu(),
            "poly_rot":   self._poly_rot.detach().cpu(),
            "poly_scale": self._poly_scale.detach().cpu(),
            "D":          self.D,
        }
        torch.save(state, os.path.join(path, "poly_deformation.pth"))
        torch.save(self._deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, "deformation_accum.pth"))

    def load_model(self, path):
        ckpt_path = os.path.join(path, "poly_deformation.pth")
        if not os.path.exists(ckpt_path):
            print(f"[PolyApprox] No poly_deformation.pth found at {path}")
            return
        state = torch.load(ckpt_path, map_location="cuda")
        self._poly_pos   = nn.Parameter(state["poly_pos"].cuda())
        self._poly_rot   = nn.Parameter(state["poly_rot"].cuda())
        self._poly_scale = nn.Parameter(state["poly_scale"].cuda())
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
            {
                "poly_pos":   self._poly_pos,
                "poly_rot":   self._poly_rot,
                "poly_scale": self._poly_scale,
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
            self.active_sh_degree, self._xyz, poly_state, self._deformation_table,
            self._features_dc, self._features_rest, self._scaling, self._rotation,
            self._opacity, self.max_radii2D, xyz_gradient_accum, denom,
            opt_dict, self.spatial_lr_scale,
        ) = model_args
        self._poly_pos   = poly_state["poly_pos"]
        self._poly_rot   = poly_state["poly_rot"]
        self._poly_scale = poly_state["poly_scale"]
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom              = denom
        self.optimizer.load_state_dict(opt_dict)
