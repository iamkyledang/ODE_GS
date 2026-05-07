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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time


def _make_raster_settings(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, cam_type):
    """Shared rasterisation settings factory used by both render() and render_ode()."""
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
        )
        t_raw = viewpoint_camera.time
    else:
        raster_settings = viewpoint_camera["camera"]
        t_raw = viewpoint_camera["time"]
    return raster_settings, t_raw
def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, stage="fine", cam_type=None):
    """Alias for render_ode — kept for backward compatibility."""
    return render_ode(viewpoint_camera, pc, pipe, bg_color, scaling_modifier,
                      override_color, stage, cam_type)


def render_ode(viewpoint_camera, pc, pipe, bg_color: torch.Tensor,
               scaling_modifier: float = 1.0, override_color=None,
               stage: str = "fine", cam_type=None):
    """
    Render a dynamic scene using ODE-parameterised Gaussian primitives
    (GaussianModel).

    For the 'coarse' stage, canonical Gaussian parameters are used directly.
    For the 'fine' stage, the camera timestamp t ∈ [0, 1] is converted to
    tau ∈ [-1, 1] and GaussianModel.apply_ode_deformation(tau) is called to
    obtain deformed means, log-scales, and quaternions, which are then passed
    to the standard 3DGS rasterizer.  The rendering pipeline itself is
    identical to render().
    """
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    raster_settings, t_raw = _make_raster_settings(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier, cam_type
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if "coarse" in stage:
        means3D_final   = pc.get_xyz
        scales_final    = pc._scaling
        rotations_final = pc._rotation
    elif "fine" in stage:
        tau = float(2.0 * t_raw - 1.0)
        means3D_final, scales_final, rotations_final = pc.apply_ode_deformation(tau)
    else:
        raise NotImplementedError(f"Unknown stage: '{stage}'")

    means2D     = screenspace_points
    opacity     = pc._opacity
    shs         = pc.get_features
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp   = pc.get_covariance(scaling_modifier)
        scales_out      = None
        rotations_out   = None
    else:
        scales_out    = pc.scaling_activation(scales_final)
        rotations_out = pc.rotation_activation(rotations_final)

    opacity_out = pc.opacity_activation(opacity)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=shs if colors_precomp is None else None,
        colors_precomp=colors_precomp,
        opacities=opacity_out,
        scales=scales_out,
        rotations=rotations_out,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
    }

