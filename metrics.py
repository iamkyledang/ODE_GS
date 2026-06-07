#
# Evaluation and visualisation for ODE-based dynamic 4D Gaussian Splatting.
#
# Measures:
#   1. Per-frame rendering quality  : PSNR, SSIM, LPIPS-vgg, LPIPS-alex, MS-SSIM, D-SSIM
#   2. Optical flow error           : EPE between rendered-flow and GT-flow
#                                     (requires torchvision >= 0.15 RAFT estimator)
#
# Visualisations saved to <model_path>/visualizations/test/<method>/:
#   metric_psnr.png          -- per-frame PSNR over time
#   metric_ssim.png          -- per-frame SSIM over time
#   metric_lpips_alex.png    -- per-frame LPIPS-alex over time
#   flow/
#     flow_error_curve_cam{N:02d}.png    -- per-frame flow error for each camera
#     flow_error_per_camera_mean.png     -- mean flow error per camera (bar chart)
#     flow_error_camera_time_heatmap.png -- camera x time heatmap
#
# Results written to:
#   <model_path>/results.json    -- summary metrics (PSNR, SSIM, LPIPS, ...)
#                                   includes flow_EPE_mean and flow_EPE_per_camera when --eval_flow
#   <model_path>/per_view.json   -- per-frame metric values
#                                   includes flow_EPE: {cam: [epe_0, epe_1, ...]} when --eval_flow
#
# Usage:
#   python metrics.py -m output/multipleview/coffee_martini
#   python metrics.py -m output/multipleview/coffee_martini --eval_flow
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
import numpy as np
from gpu import GPU_CFG, apply_torch_global_settings, log_gpu_info
apply_torch_global_settings()
log_gpu_info()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(path):
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3].cuda()


def _sorted_pairs(renders_dir, gt_dir):
    render_names = sorted(n for n in os.listdir(renders_dir) if n.endswith('.png'))
    pairs = []
    for name in render_names:
        rp = Path(renders_dir) / name
        gp = Path(gt_dir) / name
        if gp.exists():
            pairs.append((rp, gp))
    return pairs


# ---------------------------------------------------------------------------
# Rendering-quality metrics
# ---------------------------------------------------------------------------

def compute_rendering_metrics(renders_dir, gt_dir):
    pairs = _sorted_pairs(renders_dir, gt_dir)
    if not pairs:
        raise FileNotFoundError(f"No matching image pairs in {renders_dir} and {gt_dir}")

    from lpipsPyTorch.modules.lpips import LPIPS as _LPIPS
    if GPU_CFG.lpips_on_gpu:
        lpips_vgg_net  = _LPIPS(net_type="vgg").cuda().eval()
        lpips_alex_net = _LPIPS(net_type="alex").cuda().eval()
    else:
        lpips_vgg_net  = _LPIPS(net_type="vgg").cpu().eval()
        lpips_alex_net = _LPIPS(net_type="alex").cpu().eval()

    ssims_l, psnrs_l, lpips_vgg_l, lpips_alex_l, ms_ssims_l = [], [], [], [], []
    names = []

    for render_path, gt_path in tqdm(pairs, desc="Rendering metrics"):
        with torch.no_grad():
            render_cpu = tf.to_tensor(Image.open(render_path)).unsqueeze(0)[:, :3]
            gt_cpu     = tf.to_tensor(Image.open(gt_path)).unsqueeze(0)[:, :3]
            render_gpu = render_cpu.cuda()
            gt_gpu     = gt_cpu.cuda()
            ssims_l.append(ssim(render_gpu, gt_gpu).item())
            psnrs_l.append(psnr(render_gpu, gt_gpu).item())
            ms_ssims_l.append(ms_ssim(render_gpu, gt_gpu, data_range=1, size_average=True).item())
            if GPU_CFG.lpips_on_gpu:
                lpips_vgg_l.append(lpips_vgg_net(render_gpu, gt_gpu).item())
                lpips_alex_l.append(lpips_alex_net(render_gpu, gt_gpu).item())
            else:
                del render_gpu, gt_gpu
                torch.cuda.empty_cache()
                lpips_vgg_l.append(lpips_vgg_net(render_cpu, gt_cpu).item())
                lpips_alex_l.append(lpips_alex_net(render_cpu, gt_cpu).item())
                del render_cpu, gt_cpu
            names.append(Path(render_path).stem)

    del lpips_vgg_net, lpips_alex_net
    if GPU_CFG.lpips_on_gpu:
        torch.cuda.empty_cache()
    dssims_l = [(1.0 - v) / 2.0 for v in ms_ssims_l]

    summary = {
        "PSNR":       float(np.mean(psnrs_l)),
        "SSIM":       float(np.mean(ssims_l)),
        "LPIPS-vgg":  float(np.mean(lpips_vgg_l)),
        "LPIPS-alex": float(np.mean(lpips_alex_l)),
        "MS-SSIM":    float(np.mean(ms_ssims_l)),
        "D-SSIM":     float(np.mean(dssims_l)),
    }
    per_view = {
        "PSNR":       dict(zip(names, psnrs_l)),
        "SSIM":       dict(zip(names, ssims_l)),
        "LPIPS-vgg":  dict(zip(names, lpips_vgg_l)),
        "LPIPS-alex": dict(zip(names, lpips_alex_l)),
        "MS-SSIM":    dict(zip(names, ms_ssims_l)),
        "D-SSIM":     dict(zip(names, dssims_l)),
    }
    return summary, per_view


# ---------------------------------------------------------------------------
# Optical flow error (EPE via RAFT)
# ---------------------------------------------------------------------------

def _build_raft():
    """Return (raft_model, transforms) or (None, None) if unavailable."""
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    except ImportError:
        print("[flow] torchvision RAFT not available — skipping flow evaluation.")
        return None, None
    weights = Raft_Large_Weights.DEFAULT
    raft = raft_large(weights=weights).cuda().eval()
    return raft, weights.transforms()


def _infer_camera_groups(pairs):
    """
    Group frame pairs by camera.

    First tries to read cameras_metadata.json from the renders directory
    (written by render.py).  Falls back to cameras.json at the scene root,
    then to a single 'cam00' group.

    Returns:
        dict[str, list[tuple[Path, Path]]] -- {cam_name: [(render, gt), ...]}
    """
    renders_dir = pairs[0][0].parent
    cam_meta_path = renders_dir / "cameras_metadata.json"

    if cam_meta_path.exists():
        try:
            with open(cam_meta_path) as f:
                cam_meta = json.load(f)
            groups: dict = {}
            for rp, gp in pairs:
                cam_name = cam_meta.get(rp.name, "cam00")
                groups.setdefault(cam_name, []).append((rp, gp))
            if groups:
                return groups
        except Exception:
            pass

    # Fallback: try cameras.json at the scene root
    parent = pairs[0][0].parent.parent  # .../ours_N/renders -> .../ours_N
    cameras_json = parent.parent.parent / "cameras.json"
    if cameras_json.exists():
        try:
            with open(cameras_json) as f:
                cam_data = json.load(f)
            if isinstance(cam_data, list):
                n_cams = len({c["id"] for c in cam_data})
                if n_cams > 1:
                    n_frames = len(pairs) // n_cams
                    groups = {}
                    for cam_idx in range(n_cams):
                        cam_name = f"cam{cam_idx:02d}"
                        groups[cam_name] = pairs[cam_idx * n_frames:(cam_idx + 1) * n_frames]
                    return groups
        except Exception:
            pass

    return {"cam00": pairs}


def compute_flow_error(renders_dir, gt_dir):
    """
    Compare optical flow between consecutive rendered and GT frame pairs.

    Returns:
        mean_error: float or None
        per_frame:  list[float]
        per_camera: dict[str, list[float]]
    """
    raft, tfm = _build_raft()
    if raft is None:
        return None, [], {}

    pairs = _sorted_pairs(renders_dir, gt_dir)
    if len(pairs) < 2:
        return None, [], {}

    cam_groups = _infer_camera_groups(pairs)
    per_camera_errors = {}
    all_errors = []

    for cam_name, cam_pairs in cam_groups.items():
        cam_errors = []
        for i in tqdm(range(len(cam_pairs) - 1), desc=f"Flow [{cam_name}]"):
            r_a = _to_tensor(cam_pairs[i][0]);      r_b = _to_tensor(cam_pairs[i + 1][0])
            g_a = _to_tensor(cam_pairs[i][1]);      g_b = _to_tensor(cam_pairs[i + 1][1])
            r_a_t, r_b_t = tfm(r_a, r_b)
            g_a_t, g_b_t = tfm(g_a, g_b)
            with torch.no_grad():
                flow_r = raft(r_a_t, r_b_t)[-1]
                flow_g = raft(g_a_t, g_b_t)[-1]
            err = (flow_r - flow_g).norm(dim=1).mean().item()  # EPE
            cam_errors.append(err)
            all_errors.append(err)
            if GPU_CFG.metrics_cache_clear == "per_image":
                del r_a, r_b, g_a, g_b, flow_r, flow_g
                torch.cuda.empty_cache()
        if GPU_CFG.metrics_cache_clear == "per_camera":
            torch.cuda.empty_cache()  # flush once per camera, not per frame
        per_camera_errors[cam_name] = cam_errors

    mean_err = float(np.mean(all_errors)) if all_errors else None
    return mean_err, all_errors, per_camera_errors


def save_flow_visualizations(per_camera_errors, out_dir):
    """
    Save optical flow error visualisations to out_dir/:
      flow_error_curve_cam{N:02d}.png       -- per-frame error curve per camera
      flow_error_per_camera_mean.png        -- mean error per camera (bar chart)
      flow_error_camera_time_heatmap.png    -- camera x time heatmap
    """
    if not per_camera_errors:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_names    = list(per_camera_errors.keys())
    all_flat     = [e for errs in per_camera_errors.values() for e in errs]
    global_mean  = float(np.mean(all_flat)) if all_flat else 0.0

    # 1. Per-camera error curves
    _MAX_FIG_W = 200  # inches cap — avoids Pillow's 2^16 px limit at 150 dpi
    for cam_name, errors in per_camera_errors.items():
        if not errors:
            continue
        cam_mean = float(np.mean(errors))
        fig, ax = plt.subplots(figsize=(min(_MAX_FIG_W, max(8, len(errors) // 4)), 4))
        ax.plot(range(len(errors)), errors, color="steelblue", linewidth=1.2,
                marker="o", markersize=3)
        ax.axhline(cam_mean, color="red", linestyle="--",
                   label=f"mean = {cam_mean:.4f}")
        ax.set_xlabel("Frame-pair index")
        ax.set_ylabel("Optical flow EPE (px)")
        ax.set_title(f"Optical Flow Error — {cam_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"flow_error_curve_{cam_name}.png", dpi=150)
        plt.close(fig)

    # 2. Per-camera mean bar chart
    cam_means = [
        float(np.mean(per_camera_errors[c])) if per_camera_errors[c] else 0.0
        for c in cam_names
    ]
    fig, ax = plt.subplots(figsize=(min(_MAX_FIG_W, max(6, len(cam_names) * 1.2)), 5))
    ax.bar(cam_names, cam_means, color="steelblue", alpha=0.85, edgecolor="white")
    ax.axhline(global_mean, color="red", linestyle="--",
               label=f"global mean = {global_mean:.4f}")
    ax.set_xlabel("Camera pose")
    ax.set_ylabel("Mean optical flow EPE (px)")
    ax.set_title("Per-Camera Mean Optical Flow Error")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "flow_error_per_camera_mean.png", dpi=150)
    plt.close(fig)

    # 3. Camera x time heatmap
    max_frames = max(len(v) for v in per_camera_errors.values())
    heatmap = np.full((len(cam_names), max_frames), np.nan)
    for i, cam_name in enumerate(cam_names):
        errs = per_camera_errors[cam_name]
        heatmap[i, :len(errs)] = errs

    fig_h = max(2, len(cam_names) * 0.6)
    fig_w = min(_MAX_FIG_W, max(10, max_frames * 0.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.nanpercentile(heatmap, 99)) if not np.all(np.isnan(heatmap)) else 1.0
    im = ax.imshow(heatmap, aspect="auto", cmap="hot_r", vmin=0, vmax=vmax,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Optical flow EPE (px)")
    ax.set_yticks(range(len(cam_names)))
    ax.set_yticklabels(cam_names, fontsize=8)
    ax.set_xlabel("Frame-pair index")
    ax.set_ylabel("Camera pose")
    ax.set_title("Optical Flow Error — Camera × Time")
    fig.tight_layout()
    fig.savefig(out_dir / "flow_error_camera_time_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"[flow] Visualisations saved to {out_dir}")


# ---------------------------------------------------------------------------
# Per-camera rendering metrics (for train-set rendering_checkpoint.json)
# ---------------------------------------------------------------------------

def compute_rendering_metrics_per_camera(renders_dir, gt_dir):
    """
    Compute rendering metrics (PSNR, SSIM, LPIPS, MS-SSIM) per camera.

    Requires cameras_metadata.json in renders_dir to identify per-camera frames.

    Returns:
        completed_cameras: list[str]
        per_cam_results:   dict[str, dict]  {cam_name: {metric: [values...]}}
    """
    renders_dir = Path(renders_dir)
    gt_dir = Path(gt_dir)

    cam_meta_path = renders_dir / "cameras_metadata.json"
    if not cam_meta_path.exists():
        return [], {}

    with open(cam_meta_path) as f:
        cam_meta = json.load(f)

    # Group render/gt pairs by camera (preserve sort order within each camera)
    cam_groups: dict = {}
    for fname in sorted(cam_meta.keys()):
        cam_name = cam_meta[fname]
        rp = renders_dir / fname
        gp = gt_dir / fname
        if rp.exists() and gp.exists():
            cam_groups.setdefault(cam_name, []).append((rp, gp))

    if not cam_groups:
        return [], {}

    from lpipsPyTorch.modules.lpips import LPIPS as _LPIPS
    if GPU_CFG.lpips_on_gpu:
        lpips_vgg_net  = _LPIPS(net_type="vgg").cuda().eval()
        lpips_alex_net = _LPIPS(net_type="alex").cuda().eval()
    else:
        lpips_vgg_net  = _LPIPS(net_type="vgg").cpu().eval()
        lpips_alex_net = _LPIPS(net_type="alex").cpu().eval()

    per_cam_results: dict = {}
    completed_cameras: list = []

    for cam_name in sorted(cam_groups.keys()):
        cam_pairs = cam_groups[cam_name]
        psnrs, ssims_l, lpips_vgg_l, lpips_alex_l, ms_ssims_l = [], [], [], [], []

        for rp, gp in tqdm(cam_pairs, desc=f"Metrics [{cam_name}]"):
            with torch.no_grad():
                render_cpu = tf.to_tensor(Image.open(rp)).unsqueeze(0)[:, :3]
                gt_cpu     = tf.to_tensor(Image.open(gp)).unsqueeze(0)[:, :3]
                render_gpu = render_cpu.cuda()
                gt_gpu     = gt_cpu.cuda()
                ssims_l.append(ssim(render_gpu, gt_gpu).item())
                psnrs.append(psnr(render_gpu, gt_gpu).item())
                ms_ssims_l.append(
                    ms_ssim(render_gpu, gt_gpu, data_range=1, size_average=True).item()
                )
                if GPU_CFG.lpips_on_gpu:
                    lpips_vgg_l.append(lpips_vgg_net(render_gpu, gt_gpu).item())
                    lpips_alex_l.append(lpips_alex_net(render_gpu, gt_gpu).item())
                else:
                    del render_gpu, gt_gpu
                    torch.cuda.empty_cache()
                    lpips_vgg_l.append(lpips_vgg_net(render_cpu, gt_cpu).item())
                    lpips_alex_l.append(lpips_alex_net(render_cpu, gt_cpu).item())

        per_cam_results[cam_name] = {
            "PSNR":      psnrs,
            "SSIM":      ssims_l,
            "LPIPS-vgg": lpips_vgg_l,
            "LPIPS-alex": lpips_alex_l,
            "MS-SSIM":   ms_ssims_l,
        }
        completed_cameras.append(cam_name)

    del lpips_vgg_net, lpips_alex_net
    if GPU_CFG.lpips_on_gpu:
        torch.cuda.empty_cache()
    return completed_cameras, per_cam_results


# ---------------------------------------------------------------------------
# Flow visualisations — per-camera subdirectory variant (for train set)
# ---------------------------------------------------------------------------

def save_flow_visualizations_per_cam_subdirs(per_camera_errors, out_dir):
    """
    Save optical flow error visualisations using per-camera sub-directories.

    Layout (mirrors the existing output/…/visualizations/train structure):
      out_dir/
        cam00/flow_error_curve_cam00.png
        cam01/flow_error_curve_cam01.png
        ...
        flow_error_per_camera_mean.png
        flow_error_camera_time_heatmap.png
    """
    if not per_camera_errors:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_names   = list(per_camera_errors.keys())
    all_flat    = [e for errs in per_camera_errors.values() for e in errs]
    global_mean = float(np.mean(all_flat)) if all_flat else 0.0

    # 1. Per-camera error curves — each in its own sub-directory
    _MAX_FIG_W = 200  # inches cap — avoids Pillow's 2^16 px limit at 150 dpi
    for cam_name, errors in per_camera_errors.items():
        if not errors:
            continue
        cam_dir = out_dir / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)

        cam_mean = float(np.mean(errors))
        fig, ax = plt.subplots(figsize=(min(_MAX_FIG_W, max(8, len(errors) // 4)), 4))
        ax.plot(range(len(errors)), errors, color="steelblue", linewidth=1.2,
                marker="o", markersize=3)
        ax.axhline(cam_mean, color="red", linestyle="--",
                   label=f"mean = {cam_mean:.4f}")
        ax.set_xlabel("Frame-pair index")
        ax.set_ylabel("Optical flow EPE (px)")
        ax.set_title(f"Optical Flow Error — {cam_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(cam_dir / f"flow_error_curve_{cam_name}.png", dpi=150)
        plt.close(fig)

    # 2. Per-camera mean bar chart (flat in out_dir)
    cam_means = [
        float(np.mean(per_camera_errors[c])) if per_camera_errors[c] else 0.0
        for c in cam_names
    ]
    fig, ax = plt.subplots(figsize=(max(6, len(cam_names) * 1.2), 5))
    ax.bar(cam_names, cam_means, color="steelblue", alpha=0.85, edgecolor="white")
    ax.axhline(global_mean, color="red", linestyle="--",
               label=f"global mean = {global_mean:.4f}")
    ax.set_xlabel("Camera pose")
    ax.set_ylabel("Mean optical flow EPE (px)")
    ax.set_title("Per-Camera Mean Optical Flow Error")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "flow_error_per_camera_mean.png", dpi=150)
    plt.close(fig)

    # 3. Camera × time heatmap (flat in out_dir)
    max_frames = max(len(v) for v in per_camera_errors.values())
    heatmap = np.full((len(cam_names), max_frames), np.nan)
    for i, cam_name in enumerate(cam_names):
        errs = per_camera_errors[cam_name]
        heatmap[i, :len(errs)] = errs

    fig_h = max(2, len(cam_names) * 0.6)
    fig_w = min(_MAX_FIG_W, max(10, max_frames * 0.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.nanpercentile(heatmap, 99)) if not np.all(np.isnan(heatmap)) else 1.0
    im = ax.imshow(heatmap, aspect="auto", cmap="hot_r", vmin=0, vmax=vmax,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Optical flow EPE (px)")
    ax.set_yticks(range(len(cam_names)))
    ax.set_yticklabels(cam_names, fontsize=8)
    ax.set_xlabel("Frame-pair index")
    ax.set_ylabel("Camera pose")
    ax.set_title("Optical Flow Error — Camera × Time")
    fig.tight_layout()
    fig.savefig(out_dir / "flow_error_camera_time_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"[flow] Per-camera visualisations saved to {out_dir}")



# ---------------------------------------------------------------------------
# Per-frame metric time plots (PSNR / SSIM / LPIPS-alex)
# ---------------------------------------------------------------------------

def save_metric_plots(per_view, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not available — skipping metric plots.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("PSNR",       "PSNR (dB)",              "steelblue"),
        ("SSIM",       "SSIM",                   "seagreen"),
        ("LPIPS-alex", "LPIPS-alex (lower=better)", "tomato"),
    ]

    # Cap figure width: at 150 dpi Pillow rejects images wider than 2^16 px
    _MAX_FIG_W = 200  # inches → 30 000 px @ 150 dpi, well below the 65 535 limit
    for metric, ylabel, color in plot_specs:
        if metric not in per_view:
            continue
        values = list(per_view[metric].values())
        names  = list(per_view[metric].keys())
        x      = range(len(values))
        fig, ax = plt.subplots(figsize=(min(_MAX_FIG_W, max(8, len(values) // 4)), 4))
        ax.plot(x, values, color=color, linewidth=1.2, marker="o", markersize=2)
        ax.axhline(float(np.mean(values)), color="black", linestyle="--", linewidth=0.8,
                   label=f"mean = {np.mean(values):.4f}")
        ax.set_xlabel("Frame index")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Per-frame {metric}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        step = max(1, len(names) // 20)
        ax.set_xticks(list(x)[::step])
        ax.set_xticklabels(names[::step], rotation=45, ha="right", fontsize=6)
        fig.tight_layout()
        safe = metric.replace("-", "_").replace(" ", "_").lower()
        fig.savefig(out_dir / f"metric_{safe}.png", dpi=150)
        plt.close(fig)

    print(f"[viz] Metric plots saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main evaluate()
# ---------------------------------------------------------------------------

def evaluate(model_paths, eval_flow=False, model_type=None):
    for scene_dir in model_paths:
        if model_type is not None:
            scene_dir = os.path.join(scene_dir, model_type)

        if not os.path.isdir(scene_dir):
            print(
                f"[ERROR] Model directory not found: {scene_dir!r}\n"
                f"        Ensure training completed and --model_type matches the "
                f"subfolder name. Skipping."
            )
            continue

        print(f"\n{'='*60}\nScene: {scene_dir}\n{'='*60}")

        test_dir  = Path(scene_dir) / "test"
        viz_root  = Path(scene_dir) / "visualizations"
        full_dict = {}
        pv_dict   = {}

        # ------------------------------------------------------------------
        # Test-set evaluation — full metrics written to results.json
        # ------------------------------------------------------------------
        if test_dir.exists():
            for method in sorted(os.listdir(test_dir)):
                method_dir  = test_dir / method
                renders_dir = method_dir / "renders"
                gt_dir      = method_dir / "gt"
                if not renders_dir.exists() or not gt_dir.exists():
                    continue

                print(f"\n  [Test] Method: {method}")
                test_viz = viz_root / "test" / method
                test_viz.mkdir(parents=True, exist_ok=True)

                try:
                    test_summary, test_per_view = compute_rendering_metrics(
                        renders_dir, gt_dir
                    )
                    print("\n  Test metrics summary:")
                    for k, v in test_summary.items():
                        print(f"    {k:12s}: {v:.5f}")
                    # Populate results immediately so viz/flow errors cannot lose metrics
                    full_dict[f"test_{method}"] = dict(test_summary)
                    pv_dict[f"test_{method}"]   = dict(test_per_view)
                except Exception as e:
                    print(f"  [test metrics error] {e}")
                else:
                    try:
                        save_metric_plots(test_per_view, test_viz)
                    except Exception as e:
                        print(f"  [test viz warning] {e}")
                    if eval_flow:
                        try:
                            mean_flow, per_frame, per_camera = compute_flow_error(
                                renders_dir, gt_dir
                            )
                            if mean_flow is not None:
                                full_dict[f"test_{method}"]["flow_EPE_mean"] = mean_flow
                                full_dict[f"test_{method}"]["flow_EPE_per_camera"] = {
                                    cam: float(np.mean(errs))
                                    for cam, errs in per_camera.items() if errs
                                }
                                flow_pv = {
                                    cam: [float(e) for e in errs]
                                    for cam, errs in per_camera.items()
                                }
                                pv_dict[f"test_{method}"]["flow_EPE"] = flow_pv
                                print(f"    flow_EPE    : {mean_flow:.5f}")
                                save_flow_visualizations_per_cam_subdirs(
                                    per_camera, test_viz / "flow"
                                )
                        except Exception as e:
                            print(f"  [test flow error] {e}")

        # Write JSON results
        with open(os.path.join(scene_dir, "results.json"), "w") as fp:
            json.dump(full_dict, fp, indent=2)
        with open(os.path.join(scene_dir, "per_view.json"), "w") as fp:
            json.dump(pv_dict, fp, indent=2)
        print(f"\n  Results -> {scene_dir}/results.json")
        print(f"  Per-view -> {scene_dir}/per_view.json")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(
        description="Evaluation and visualisation for ODE 4D Gaussian Splatting"
    )
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str,
                        help="One or more trained model directories.")
    parser.add_argument("--eval_flow", action="store_true", default=True,
                        help="Evaluate optical flow error (requires RAFT in torchvision >= 0.15).")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Model-type subfolder appended to each model path "
                             "(e.g. real_ode or complex_ode).")
    args = parser.parse_args()

    evaluate(
        args.model_paths,
        eval_flow=args.eval_flow,
        model_type=args.model_type,
    )
