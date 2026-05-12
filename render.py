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
import copy
import imageio
import json
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ODEModelParams
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_ode


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
    else:
        raise ValueError(f"Unknown --model_class: {model_class!r}")
    return _GM
from time import time
import threading
import concurrent.futures


# ---------------------------------------------------------------------------
# GPU capability detection
# ---------------------------------------------------------------------------

def _is_high_memory_gpu() -> bool:
    """Return True when a high-VRAM GPU (RTX 4090, 24 GB) is detected."""
    if not torch.cuda.is_available():
        return False
    return "4090" in torch.cuda.get_device_name(0).lower()

_HIGH_MEM: bool = _is_high_memory_gpu()

def lerp(a, b, alpha):
    return (1.0 - alpha) * a + alpha * b

def build_interpolated_view(view0, view1, render_time, pose_alpha):
    view = copy.deepcopy(view0)

    # interpolate translation
    if hasattr(view0, "T") and hasattr(view1, "T"):
        view.T = lerp(view0.T, view1.T, pose_alpha)

    # interpolate rotation
    if hasattr(view0, "R") and hasattr(view1, "R"):
        R = lerp(view0.R, view1.R, pose_alpha)

        # optional re-orthonormalization
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        view.R = R

    # set normalized time
    if hasattr(view, "time"):
        view.time = render_time
    elif hasattr(view, "fid"):
        view.fid = render_time
    else:
        print("Could not find time field. Available attrs:")
        print(dir(view))
        raise AttributeError("Camera/view object has no 'time' or 'fid' field.")

    return view


def build_custom_view_from_template(template_view, render_time, custom_R, custom_T):
    view = copy.deepcopy(template_view)

    R = np.array(custom_R, dtype=np.float32).reshape(3, 3)
    T = np.array(custom_T, dtype=np.float32)

    # overwrite pose
    if hasattr(view, "R"):
        view.R = R
    else:
        raise AttributeError("Template view has no attribute 'R'")

    if hasattr(view, "T"):
        view.T = T
    else:
        raise AttributeError("Template view has no attribute 'T'")

    # overwrite time
    if hasattr(view, "time"):
        view.time = float(render_time)
    elif hasattr(view, "fid"):
        view.fid = float(render_time)
    else:
        print("Available attributes on view:")
        print(dir(view))
        raise AttributeError("Template view has no 'time' or 'fid' attribute")

    return view

def render_custom_image(model_path, iteration, views, gaussians, pipeline, background,
                        cam_type, render_time, template_index, custom_R, custom_T, source_set,
                        render_fn=None):
    if render_fn is None:
        render_fn = render
    out_dir = os.path.join(model_path, "custom_renders", "ours_{}".format(iteration))
    makedirs(out_dir, exist_ok=True)

    if render_time is None:
        raise ValueError("--render_time is required when using --render_custom")
    if not (0.0 <= render_time <= 1.0):
        raise ValueError(f"render_time must be in [0,1], got {render_time}")

    if custom_R is None or len(custom_R) != 9:
        raise ValueError("--custom_R must contain exactly 9 numbers")
    if custom_T is None or len(custom_T) != 3:
        raise ValueError("--custom_T must contain exactly 3 numbers")

    if template_index < 0 or template_index >= len(views):
        raise ValueError(f"template_index={template_index} out of range [0, {len(views)-1}]")

    template_view = views[template_index]
    custom_view = build_custom_view_from_template(template_view, render_time, custom_R, custom_T)

    with torch.no_grad():
        rendering = render_fn(custom_view, gaussians, pipeline, background, cam_type=cam_type)["render"]

    R_str = "_".join([f"{x:.3f}" for x in custom_R])
    T_str = "_".join([f"{x:.3f}" for x in custom_T])

    filename = (
        f"{source_set}_template{template_index:03d}"
        f"_t{render_time:.3f}"
        f"_R_{R_str}"
        f"_T_{T_str}.png"
    )
    out_path = os.path.join(out_dir, filename)

    torchvision.utils.save_image(rendering.cpu(), out_path)
    print(f"Saved custom render to: {out_path}")

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, render_fn=None):
    if render_fn is None:
        render_fn = render
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    render_images = []
    print("point nums:", gaussians._xyz.shape[0])

    cam_names_dict = {}

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:
            time1 = time()

        with torch.no_grad():
            rendering = render_fn(view, gaussians, pipeline, background, cam_type=cam_type)["render"]

        render_cpu = rendering.detach().cpu()
        torchvision.utils.save_image(
            render_cpu,
            os.path.join(render_path, f"{idx:05d}.png")
        )

        frame = (255 * np.clip(render_cpu.numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0)
        frame = np.ascontiguousarray(frame)
        render_images.append(frame)

        # Record camera name for metadata
        cam_names_dict[f"{idx:05d}.png"] = getattr(view, "image_name", f"cam{idx:02d}")

        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt = view["image"]

            gt_cpu = gt.detach().cpu() if torch.is_tensor(gt) else gt.cpu()
            torchvision.utils.save_image(
                gt_cpu,
                os.path.join(gts_path, f"{idx:05d}.png")
            )
            del gt_cpu

        del rendering, render_cpu
        if not _HIGH_MEM:
            torch.cuda.empty_cache()

    if _HIGH_MEM:
        torch.cuda.empty_cache()  # flush once after the full render loop
    time2 = time()
    print("FPS:", (len(views) - 1) / (time2 - time1))

    # Save cameras metadata for train/test splits
    if name in ["train", "test"]:
        meta_path = os.path.join(render_path, "cameras_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(cam_names_dict, f, indent=2)

    imageio.mimwrite(
        os.path.join(model_path, name, "ours_{}".format(iteration), "video_rgb.mp4"),
        render_images,
        fps=30,
        macro_block_size=1
    )

def render_sets(dataset: ModelParams, hyperparam, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, skip_video: bool,
                render_custom=False, render_time=None, source_set="test",
                template_index=0, custom_R=None, custom_T=None,
                model_type=None, model_class=None):
    with torch.no_grad():
        # Append model-type subfolder so different variants live in separate directories
        if model_type is not None:
            dataset.model_path = os.path.join(dataset.model_path, model_type)

        if not os.path.isdir(dataset.model_path):
            raise FileNotFoundError(
                f"Model directory not found: {dataset.model_path!r}\n"
                f"Ensure training completed successfully and that --model_type "
                f"matches the subfolder created during training "
                f"(e.g. real_ode or complex_ode)."
            )

        GaussianModelClass = _get_gaussian_model_class(model_class)
        gaussians = GaussianModelClass(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type = scene.dataset_type
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_fn = render_ode

        if render_custom:
            if source_set == "train":
                views = scene.getTrainCameras()
            elif source_set == "test":
                views = scene.getTestCameras()
            elif source_set == "video":
                views = scene.getVideoCameras()
            else:
                raise ValueError(f"Unknown source_set: {source_set}")

            render_custom_image(
                dataset.model_path,
                scene.loaded_iter,
                views,
                gaussians,
                pipeline,
                background,
                cam_type,
                render_time,
                template_index,
                custom_R,
                custom_T,
                source_set,
                render_fn
            )
            return

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
                render_fn
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
                render_fn
            )

        if not skip_video:
            render_set(
                dataset.model_path,
                "video",
                scene.loaded_iter,
                scene.getVideoCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
                render_fn
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ODEModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--model_type", type=str, default=None,
                        help="Model-type subfolder to append to model_path "
                             "(e.g. real_ode or complex_ode). "
                             "Must match the subfolder created during training.")
    parser.add_argument("--model_class", type=str, default="ode",
                        help="Gaussian model class: ode (default), deformable_mlp, "
                             "deformable_hexplane_mlp, fourier_approx, polynomial_approx.")

    parser.add_argument("--render_custom", action="store_true", default=False)
    parser.add_argument("--render_time", type=float, default=0.0)
    parser.add_argument("--source_set", type=str, default="test")
    parser.add_argument("--template_index", type=int, default=0)
    parser.add_argument("--custom_R", nargs=9, type=float, default=None)
    parser.add_argument("--custom_T", nargs=3, type=float, default=None)
    
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    if not hasattr(args, "render_custom"):
        args.render_custom = False
    if not hasattr(args, "render_time"):
        args.render_time = 0.0
    if not hasattr(args, "source_set"):
        args.source_set = "test"
    if not hasattr(args, "template_index"):
        args.template_index = 0
    if not hasattr(args, "custom_R"):
        args.custom_R = None
    if not hasattr(args, "custom_T"):
        args.custom_T = None
    if not hasattr(args, "model_type"):
        args.model_type = None
    if not hasattr(args, "model_class"):
        args.model_class = "ode"

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
        args.render_custom,
        args.render_time,
        args.source_set,
        args.template_index,
        args.custom_R,
        args.custom_T,
        args.model_type,
        args.model_class
    )