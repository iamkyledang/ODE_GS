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
import numpy as np
import random
import os, sys

# ── Unified GPU configuration (must be imported before first CUDA allocation).
# gpu.py detects the hardware tier (LOW / HIGH / ULTRA) and provides all
# hardware-specific tuning knobs in one place.
from gpu import GPU_CFG, apply_torch_global_settings, apply_torch_backend_settings, log_gpu_info
# Set PYTORCH_CUDA_ALLOC_CONF env-var NOW (must be before the first CUDA memory
# allocation).  Backend flags (TF32, cuDNN benchmark) are applied AFTER
# safe_state() initialises the CUDA context to avoid triggering a sticky CUDA
# error on older PyTorch / Python 3.7 builds.
import os
if GPU_CFG.cuda_alloc_conf:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", GPU_CFG.cuda_alloc_conf)

import gc
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render_ode, network_gui
import sys
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state


def _get_gaussian_model_class(model_class: str):
    """
    Return the GaussianModel class for the requested model variant.

    Values: ode (default), deformable_mlp, deformable_hexplane_mlp,
            fourier_approx, polynomial_approx.
    """
    if model_class in (None, "", "ode"):
        from scene.gaussian_model import GaussianModel as _GM
    elif model_class == "deformable_mlp":
        from baselines.deformable_MLP import GaussianModel as _GM
    elif model_class == "deformable_hexplane_mlp":
        from baselines.deformable_hexplane_MLP import GaussianModel as _GM
    elif model_class == "fourier_approx":
        from baselines.fourier_approx import GaussianModel as _GM
    elif model_class == "polynomial_approx":
        from baselines.polynomial_approx import GaussianModel as _GM
    elif model_class == "neural_ode":
        from baselines.neural_ode import GaussianModel as _GM
    else:
        raise ValueError(f"Unknown --model_class: {model_class!r}")
    return _GM
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, ODEModelParams, ODEOptimizationParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer,
                         render_fn=None):
    """One training stage (coarse or fine)."""
    if render_fn is None:
        render_fn = render_ode
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()


    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack, batch_size=batch_size, sampler=sampler,
                num_workers=GPU_CFG.num_workers, collate_fn=list,
                pin_memory=GPU_CFG.pin_memory,
                persistent_workers=GPU_CFG.persistent_workers,
            )
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack, batch_size=batch_size, shuffle=True,
                num_workers=GPU_CFG.num_workers, collate_fn=list,
                pin_memory=GPU_CFG.pin_memory,
                persistent_workers=GPU_CFG.persistent_workers,
            )
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
                            # 
    count = 0
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    net_image = render_fn(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(
                        viewpoint_stack, batch_size=opt.batch_size, shuffle=True,
                        num_workers=GPU_CFG.num_workers, collate_fn=list,
                        pin_memory=GPU_CFG.pin_memory,
                        persistent_workers=GPU_CFG.persistent_workers,
                    )
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Free list copies immediately — combined tensors hold all needed data.
        del images, gt_images, radii_list, visibility_filter_list

        # Loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        

        loss = Ll1
        if stage == "fine":
            if hyper.lambda_ode != 0:
                loss += hyper.lambda_ode * gaussians.compute_ode_regulation()
        if opt.lambda_dssim != 0:
            if GPU_CFG.ssim_downsample_factor < 1.0:
                # Downsample before SSIM to reduce its activation-buffer VRAM
                # footprint.  Flush the allocator cache first so SSIM gets
                # fresh unfragmented blocks.
                torch.cuda.empty_cache()
                _img_ssim = torch.nn.functional.interpolate(
                    image_tensor,
                    scale_factor=GPU_CFG.ssim_downsample_factor,
                    mode="bilinear", align_corners=False,
                )
                _gt_ssim = torch.nn.functional.interpolate(
                    gt_image_tensor[:, :3, :, :],
                    scale_factor=GPU_CFG.ssim_downsample_factor,
                    mode="bilinear", align_corners=False,
                )
                ssim_loss = ssim(_img_ssim, _gt_ssim)
                del _img_ssim, _gt_ssim
            else:
                ssim_loss = ssim(image_tensor, gt_image_tensor[:, :3, :, :])
            loss += opt.lambda_dssim * (1.0 - ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss

        # Pre-backward flush: on low-VRAM GPUs the allocator often holds
        # fragmented cached blocks that prevent backward from finding a
        # contiguous slab for gradient buffers.
        if GPU_CFG.cache_clear_before_backward:
            torch.cuda.empty_cache()

        loss.backward()

        # Release large forward-pass tensors immediately after backward to free
        # activation buffers before densification allocates new ones.
        del image_tensor, gt_image_tensor
        if GPU_CFG.aggressive_cache_clear:
            torch.cuda.empty_cache()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Periodic Python GC + CUDA cache flush.
            # Frequency is controlled by GPU_CFG.gc_interval (50 on LOW,
            # 200 on HIGH, 500 on ULTRA) — frees Python-side cyclic garbage
            # that holds CUDA tensors before densification can spike VRAM.
            if iteration % GPU_CFG.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_fn, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
                # Flush after checkpoint write — saving serialises large tensors
                # which temporarily spikes allocator usage.
                torch.cuda.empty_cache()
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render_fn, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render_fn, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < GPU_CFG.max_gaussians:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                    if GPU_CFG.aggressive_cache_clear:
                        torch.cuda.empty_cache()  # free old Gaussian tensors freed by densify
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    if GPU_CFG.aggressive_cache_clear:
                        torch.cuda.empty_cache()  # immediately reclaim pruned Gaussian memory

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < GPU_CFG.max_gaussians and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    if GPU_CFG.aggressive_cache_clear:
                        torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    if GPU_CFG.aggressive_cache_clear:
                        torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    tb_writer = prepare_output_and_logger(expname)

    # Apply GPU-level densification threshold override after argument parsing.
    # On LOW-VRAM GPUs (RTX 3060): higher threshold reduces Gaussian count growth,
    #   keeping peak VRAM and CPU-side bookkeeping arrays within budget.
    # On ULTRA-VRAM GPUs (RTX PRO 6000): lower threshold enables more aggressive
    #   densification, producing denser point clouds within the large VRAM budget.
    if GPU_CFG.densify_grad_threshold_override is not None:
        opt.densify_grad_threshold_coarse    = GPU_CFG.densify_grad_threshold_override
        opt.densify_grad_threshold_fine_init = GPU_CFG.densify_grad_threshold_override
        opt.densify_grad_threshold_after     = GPU_CFG.densify_grad_threshold_override

    GaussianModelClass = _get_gaussian_model_class(getattr(args, "model_class", None))
    gaussians = GaussianModelClass(dataset.sh_degree, hyper)
    render_fn = render_ode

    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, timer,
                             render_fn=render_fn)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations, timer,
                         render_fn=render_fn)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        if GPU_CFG.is_low_vram:
            torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()  # always flush after validation
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    # ── Hardware info ────────────────────────────────────────────────────────
    log_gpu_info()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = ODEOptimizationParams(parser)
    hp = ODEModelParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--model_class", type=str, default="ode",
                        help="Gaussian model class to use: ode (default), "
                             "deformable_mlp, deformable_hexplane_mlp, "
                             "fourier_approx, polynomial_approx.")

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # ── CUDA pre-flight check ─────────────────────────────────────────────────
    # Catch Error 304 (cudaErrorOperatingSystem) early and give a clear message.
    # The most common cause in Docker is a missing --gpus all flag.
    try:
        _cuda_ok = torch.cuda.is_available()
    except Exception:
        _cuda_ok = False
    if not _cuda_ok:
        print(
            "[ERROR] CUDA device not accessible.\n"
            "  Most likely cause: Docker container started without GPU support.\n"
            "  Fix: docker run --gpus all  (or --runtime=nvidia)\n"
            "  Verify: nvidia-smi inside the container should list your GPU."
        )
        sys.exit(1)

    safe_state(args.quiet)
    # Apply TF32 / cuDNN-benchmark flags AFTER CUDA context is initialised to
    # avoid setting a sticky CUDA error on old PyTorch (Python 3.7 era).
    apply_torch_backend_settings()
    torch.cuda.empty_cache()
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
