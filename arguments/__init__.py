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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.render_process=False
        self.add_points=False
        self.extension=".png"
        self.llffhold=8
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")
class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4 # useless
        self.defor_depth = 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.opacity_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.bounds = 1.6 
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.01 # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }
        self.multires = [1, 2, 4, 8] # multi resolution of voxel grid
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds=False # cancel the deformation of Gaussians' scaling
        self.no_dr=False # cancel the deformation of Gaussians' rotations
        self.no_do=True # cancel the deformation of Gaussians' opacity
        self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False # useless
        self.grid_pe=0 # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp=False # useless
        self.apply_rotation=False # useless

        
        super().__init__(parser, "ModelHiddenParams")
        
class ODEModelParams(ParamGroup):
    """
    Hyperparameters specific to the ODE-based temporal model.
    """
    def __init__(self, parser):
        # Weight for ODE regularisation loss (trajectory + covariance)
        self.lambda_ode = 0.001
        # Weight for optical flow consistency loss (0 = disabled)
        self.lambda_flow = 0.0
        # Spatial bounding box half-extent (kept for compatibility with some dataset loaders)
        self.bounds = 1.6
        # Dummy fields kept for dataset-loader compatibility (no-op for ODE model).
        self.plane_tv_weight = 0.0
        self.time_smoothness_weight = 0.0
        self.l1_time_planes = 0.0
        # ── Deformable MLP baseline arch params ────────────────────────────
        self.deform_mlp_width = 256   # hidden width W of deformation MLP
        self.deform_mlp_depth = 8     # number of hidden layers D
        self.deform_pos_pe    = 10    # positional encoding frequencies for xyz
        self.deform_time_pe   = 4     # positional encoding frequencies for t
        # ── HexPlane + MLP baseline arch params ────────────────────────────
        self.hexplane_spatial_res = 64   # spatial grid resolution (x,y,z)
        self.hexplane_time_res    = 150  # temporal grid resolution (t)
        self.hexplane_feat_dim    = 16   # output feature dimension per plane
        self.hexplane_decode_W    = 128  # decoder MLP hidden width
        self.hexplane_decode_D    = 1    # decoder MLP hidden layers
        # ── Fourier approximation baseline arch param ───────────────────────
        self.fourier_K = 4    # number of Fourier frequency components
        # ── Polynomial approximation baseline arch param ────────────────────
        self.poly_D    = 3    # polynomial degree (terms tau^1 … tau^D); matches Gaussian-Flow traj_dim=3
        # ── Neural ODE velocity field baseline arch params ──────────────────────
        self.neural_ode_width   = 128   # velocity MLP hidden width  (evogs net_width default for dynerf)
        self.neural_ode_depth   = 1     # velocity MLP hidden layers (evogs defor_depth)
        self.neural_ode_pos_pe  = 10    # positional encoding frequencies for xyz (evogs posebase_pe)
        self.neural_ode_time_pe = 4     # positional encoding frequencies for t   (evogs timebase_pe)
        self.neural_ode_steps   = 4     # Euler integration steps
        super().__init__(parser, "ODE Model Parameters")


class ODEOptimizationParams(ParamGroup):
    """
    Optimisation parameters for the ODE model.
    Mirrors OptimizationParams with identical defaults so that the same
    config files (arguments/dnerf/*.py etc.) can be reused.
    Additional fields control the per-Gaussian ODE parameter learning rates.
    """
    def __init__(self, parser):
        self.dataloader = False
        self.zerostamp_init = False
        self.custom_sampler = None
        self.iterations = 100_000
        self.coarse_iterations = 3000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        # LR schedule spans ~2/3 of a typical 30k–100k training run;
        # both xyz and ODE params share this decay schedule.
        self.position_lr_max_steps = 30_000
        # Per-Gaussian ODE parameter learning rates.
        # The effective LR is ode_lr_init * spatial_lr_scale (~5 for indoor).
        # 0.001 gives ~5e-3 effective LR, enough for the 21 ODE params
        # (A_flat, b, x0, kappa, omega_cov) to learn meaningful dynamics.
        self.ode_lr_init = 0.001
        self.ode_lr_final = 0.0001
        self.ode_lr_delay_mult = 0.01
        # Standard Gaussian attribute learning rates.
        # Lower scaling_lr (0.001 vs vanilla 0.005) because covariance
        # evolution is handled by the kappa/omega_cov ODE params —
        # the canonical scale should be stable.
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_lpips = 0
        self.weight_constraint_init = 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.batch_size = 1
        self.add_point = False
        super().__init__(parser, "ODE Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
