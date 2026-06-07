#
# full_eval.py  —  Baseline comparison for ODE-based dynamic 4D Gaussian Splatting.
#
# Trains six temporal-deformation models on coffee_martini (MultipleView dataset)
# and evaluates them with the full metrics suite.  All method configs are declared
# inline below — no external config files are needed.
#
# Methods compared:
#   1. ode                     — proposed: 3D ODE deformation (this repo)
#   2. deformable_mlp          — per-Gaussian deformation MLP (ingra14m/Deformable-3D-Gaussians)
#   3. deformable_hexplane_mlp — HexPlane + MLP (hustvl/4DGaussians)
#   4. fourier_approx          — per-Gaussian Fourier series (raven38/EfficientDynamic3DGaussian)
#   5. polynomial_approx       — per-Gaussian polynomial (NJU-3DV/Gaussian-Flow)
#   6. neural_ode              — shared velocity MLP + Euler ODE (arnold-caleb/evogs)
#
# Usage — full comparison:
#   python full_eval.py                              # train + render + metrics, all methods
#   python full_eval.py --skip_training              # skip training (use existing checkpoints)
#   python full_eval.py --skip_rendering             # skip rendering
#   python full_eval.py --skip_metrics               # skip metrics computation
#   python full_eval.py --data_root /path/to/data    # custom data root
#
# Usage — single-method runs (train → render → metrics for ONE baseline):
#   python full_eval.py --method ode
#   python full_eval.py --method deformable_mlp
#   python full_eval.py --method deformable_hexplane_mlp
#   python full_eval.py --method fourier_approx
#   python full_eval.py --method polynomial_approx
#   python full_eval.py --method neural_ode
#   python full_eval.py --method ode --skip_training   # render + metrics only
#
# Usage — explicit subset:
#   python full_eval.py --methods ode fourier_approx polynomial_approx
#
# Hardware:
#   RTX 4090 (24 GB) is detected automatically.  batch_size and memory optimisations
#   scale up on the 4090; the script falls back to conservative settings on any
#   other GPU (e.g. RTX 3060 6 GB).
#
# All outputs are written under output/output_baselines/ (relative to cwd).
#

import os
import re
import json
import shutil
import time
import numpy as np
import torch
from argparse import ArgumentParser
from pathlib import Path
from gpu import GPU_CFG, apply_torch_global_settings, log_gpu_info
apply_torch_global_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Scene / data defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SCENE    = "coffee_martini"
DEFAULT_DATA_ROOT = "data/multipleview"
DEFAULT_OUTPUT   = "output/output_baselines"

# ─────────────────────────────────────────────────────────────────────────────
# Method configurations
# All hyperparameters are declared here as CLI arg strings; no external
# config files are used (no --configs flag).
# ─────────────────────────────────────────────────────────────────────────────

# Shared base flags for MultipleView / coffee_martini.
# All hardware-specific values (batch_size, num_workers, mem flags) come
# from GPU_CFG so the configs below are always in sync with gpu.py.
_BASE_FLAGS = (
    "--dataloader "
    f"--batch_size {GPU_CFG.batch_size} "
    "--opacity_threshold_coarse 0.005 "
    "--opacity_threshold_fine_init 0.005 "
    "--opacity_threshold_fine_after 0.005 "
    "--port 0 "  # port 0 -> OS picks a free ephemeral port; avoids EADDRINUSE on re-runs
    + (f"{GPU_CFG.cli_num_workers_flag} " if GPU_CFG.cli_num_workers_flag else "")
    + "--quiet"
)

# GPU-tier densification threshold overrides (empty string = use method defaults).
# Sourced from GPU_CFG so LOW/HIGH/ULTRA configs are always synchronized.
_GPU_MEM_FLAGS: str = GPU_CFG.cli_mem_flags

METHOD_CONFIGS = {
    # ── Proposed: ODE deformation ──────────────────────────────────────────
    # Per-Gaussian closed-form ODE (A, b, x0, kappa, omega) — no neural net.
    # Key optimizations vs vanilla defaults:
    #   - 30k iterations: matches other baselines for a fair comparison.
    #     The closed-form ODE converges faster than MLP-based methods.
    #   - ode_lr_init=0.001: effective LR ≈ ode_lr_init × spatial_lr_scale (~5).
    #     The 21 per-Gaussian ODE params start at zero; a higher LR lets them
    #     learn non-trivial dynamics within the training budget.
    #   - ode_lr_final=0.0001: ~10× decay ratio, keeps refinement stable.
    #   - position_lr_max_steps=20000: LR decays to minimum at 2/3 of training,
    #     leaving the final 10k iterations for fine-grained ODE refinement.
    #   - scaling_lr=0.001: canonical scale changes slowly because temporal
    #     covariance variation is already captured by kappa/omega_cov ODE params.
    #   - lambda_ode=0.001: regularises trajectory velocity (L_traj), covariance
    #     rotation speed (L_omega), and scale rate (L_s) from framework.tex §7.3.
    "ode": dict(
        model_class="ode",
        train_args=(
            "--model_class ode "
            "--iterations 30000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 15000 "
            "--position_lr_max_steps 20000 "
            "--ode_lr_init 0.001 "
            "--ode_lr_final 0.0001 "
            "--scaling_lr 0.001 "
            "--lambda_ode 0.001 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--plane_tv_weight 0.0 "
            "--time_smoothness_weight 0.0 "
            "--l1_time_planes 0.0 "
            + _GPU_MEM_FLAGS
            + _BASE_FLAGS
        ),
    ),

    # ── Deformable 3D Gaussians (MLP) ────────────────────────────────────
    # Based on: ingra14m/Deformable-3D-Gaussians (CVPR 2024)
    # Reference configs from arguments/__init__.py OptimizationParams:
    #   iterations=40000, warm_up=3000, scaling_lr=0.001,
    #   densify_grad_threshold=0.0007, position_lr_max_steps=30000
    # Arch params:
    #   deform_mlp_width=256, deform_mlp_depth=8, deform_pos_pe=10, deform_time_pe=4
    "deformable_mlp": dict(
        model_class="deformable_mlp",
        train_args=(
            "--model_class deformable_mlp "
            "--iterations 40000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 15000 "
            "--position_lr_max_steps 30000 "
            "--scaling_lr 0.001 "
            "--densify_grad_threshold_fine_init 0.0007 "
            "--densify_grad_threshold_after 0.0007 "
            "--lambda_ode 0.0 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--plane_tv_weight 0.0 "
            "--deform_mlp_width 256 "
            "--deform_mlp_depth 8 "
            "--deform_pos_pe 10 "
            "--deform_time_pe 4 "
            + _GPU_MEM_FLAGS
            + _BASE_FLAGS
        ),
    ),

    # ── 4D Gaussians (HexPlane + MLP) ────────────────────────────────────
    # Based on: hustvl/4DGaussians (CVPR 2024)
    # Reference configs from arguments/multipleview/default.py:
    #   iterations=15000, coarse=3000, densify_until=10000,
    #   net_width=128, defor_depth=0, multires=[1,2], kplanes=[64,64,64,150],
    #   plane_tv=0.0002, time_smooth=0.001, l1_time=0.0001, feat_dim=16
    # TV/smoothness losses enabled via lambda_ode=1.0 + compute_ode_regulation()
    "deformable_hexplane_mlp": dict(
        model_class="deformable_hexplane_mlp",
        train_args=(
            "--model_class deformable_hexplane_mlp "
            "--iterations 15000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 10000 "
            "--lambda_ode 1.0 "
            "--plane_tv_weight 0.0002 "
            "--time_smoothness_weight 0.001 "
            "--l1_time_planes 0.0001 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--hexplane_spatial_res 64 "
            "--hexplane_time_res 150 "
            "--hexplane_feat_dim 16 "
            "--hexplane_decode_W 128 "
            "--hexplane_decode_D 0 "
            + _GPU_MEM_FLAGS
            + _BASE_FLAGS
        ),
    ),

    # ── Fourier approximation ────────────────────────────────────────────
    # Based on: raven38/EfficientDynamic3DGaussian
    # Reference configs from arguments/__init__.py OptimizationParams:
    #   iterations=30000, lambda_lasso=0 (L1 reg via lambda_ode), approx_l=2 → fourier_K=2
    "fourier_approx": dict(
        model_class="fourier_approx",
        train_args=(
            "--model_class fourier_approx "
            "--iterations 30000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 15000 "
            "--lambda_ode 0.0 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--plane_tv_weight 0.0 "
            "--fourier_K 2 "
            + _GPU_MEM_FLAGS
            + _BASE_FLAGS
        ),
    ),

    # ── Polynomial approximation ─────────────────────────────────────────
    # Based on: NJU-3DV/Gaussian-Flow (poly_fourier with traj_dim=3 → degree 3)
    # Reference configs from configs/dnerf.yaml:
    #   max_steps=30000, densify_grad_threshold=0.0002
    "polynomial_approx": dict(
        model_class="polynomial_approx",
        train_args=(
            "--model_class polynomial_approx "
            "--iterations 30000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 15000 "
            "--lambda_ode 0.0 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--plane_tv_weight 0.0 "
            "--poly_D 3 "
            + _GPU_MEM_FLAGS
            + _BASE_FLAGS
        ),
    ),

    # ── Neural ODE velocity field ─────────────────────────────────────────
    # Based on: arnold-caleb/evogs (EvoGS)
    # Reference configs from arguments/dynerf/base.py + base_velocity.py:
    #   iterations=14000 (dynerf/base), coarse=3000, densify_until=10000,
    #   net_width=128, velocity field enabled, Euler integration with 4 steps
    # Velocity reg via lambda_ode=0.001 + compute_ode_regulation()
    "neural_ode": dict(
        model_class="neural_ode",
        train_args=(
            "--model_class neural_ode "
            "--iterations 14000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 10000 "
            "--lambda_ode 0.001 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--plane_tv_weight 0.0 "
            "--neural_ode_width 128 "
            "--neural_ode_depth 1 "
            "--neural_ode_pos_pe 10 "
            "--neural_ode_time_pe 4 "
            "--neural_ode_steps 4 "
            + _GPU_MEM_FLAGS
            + _BASE_FLAGS
        ),
    ),
}

ALL_METHODS = list(METHOD_CONFIGS.keys())
METRICS     = ["PSNR", "SSIM", "LPIPS-vgg", "LPIPS-alex", "MS-SSIM", "D-SSIM"]

# ─────────────────────────────────────────────────────────────────────────────
# Resume helpers — detect whether a phase has already completed
# ─────────────────────────────────────────────────────────────────────────────

def _parse_final_iteration(train_args: str) -> int:
    """Return the --iterations value from a train_args string, or 0 if absent."""
    m = re.search(r"--iterations\s+(\d+)", train_args)
    return int(m.group(1)) if m else 0


def is_training_done(method: str, scene: str, output_root: str) -> bool:
    """
    Training is complete when the final-iteration point_cloud.ply exists.
    Falls back to any checkpoint if the target iteration cannot be parsed.
    """
    out_dir    = Path(output_root) / method / scene
    cfg        = METHOD_CONFIGS[method]
    final_iter = _parse_final_iteration(cfg["train_args"])
    if final_iter > 0:
        ply = out_dir / "point_cloud" / f"iteration_{final_iter}" / "point_cloud.ply"
        return ply.exists()
    # Fallback: any iteration checkpoint present
    pc_dir = out_dir / "point_cloud"
    if not pc_dir.is_dir():
        return False
    return any(pc_dir.rglob("point_cloud.ply"))


def is_rendering_done(method: str, scene: str, output_root: str) -> bool:
    """
    Rendering is complete when test/ contains at least one ours_* subfolder
    that holds rendered PNG images.
    """
    test_dir = Path(output_root) / method / scene / "test"
    if not test_dir.is_dir():
        return False
    for ours_dir in test_dir.iterdir():
        if ours_dir.is_dir() and ours_dir.name.startswith("ours_"):
            if any(ours_dir.rglob("*.png")):
                return True
    return False


def is_metrics_done(method: str, scene: str, output_root: str) -> bool:
    """
    Metrics are done when results.json contains at least one key whose value is
    a dict (an actual metric result sub-object).  Top-level scalar fields added
    by _patch_results_json (training_time_s, render_time_s, …) are intentionally
    ignored so a partially-written file is not mistaken for complete metrics.
    """
    results = Path(output_root) / method / scene / "results.json"
    if not results.exists():
        return False
    try:
        with open(results) as f:
            data = json.load(f)
        return any(isinstance(v, dict) for v in data.values())
    except (json.JSONDecodeError, OSError):
        return False


def is_samples_done(method: str, scene: str, output_root: str) -> bool:
    """
    Sample extraction is complete when sample/ contains at least pose_1.png.
    A sentinel file sample/.done (written by sample_method) is used as the
    authoritative marker so a partial extraction is never mistaken for complete.
    """
    sample_dir = Path(output_root) / method / scene / "sample"
    return (sample_dir / ".done").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Training-time helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string (e.g. '1h 23m', '45m 07s')."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


_TRAINING_TIME_FILE = "training_time.json"
_RENDER_TIME_FILE   = "render_time.json"


def _save_training_time(method: str, scene: str, output_root: str, elapsed_s: float):
    """Persist training duration to <output_root>/<method>/<scene>/training_time.json."""
    out_dir = Path(output_root) / method / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / _TRAINING_TIME_FILE, "w") as f:
        json.dump({"seconds": round(elapsed_s, 1), "formatted": _fmt_time(elapsed_s)}, f)


def _load_training_time(method: str, scene: str, output_root: str) -> dict:
    """Return {"seconds": X, "formatted": "Xh Ym"}, or {} if not yet recorded."""
    p = Path(output_root) / method / scene / _TRAINING_TIME_FILE
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_render_time(method: str, scene: str, output_root: str, elapsed_s: float):
    """Persist rendering duration to <output_root>/<method>/<scene>/render_time.json."""
    out_dir = Path(output_root) / method / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / _RENDER_TIME_FILE, "w") as f:
        json.dump({"seconds": round(elapsed_s, 1), "formatted": _fmt_time(elapsed_s)}, f)


def _load_render_time(method: str, scene: str, output_root: str) -> dict:
    """Return {"seconds": X, "formatted": "Xh Ym"}, or {} if not yet recorded."""
    p = Path(output_root) / method / scene / _RENDER_TIME_FILE
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _patch_results_json(method: str, scene: str, output_root: str):
    """
    Inject training_time and render_time into the per-method results.json.
    Each field is written only if the corresponding *_time.json file exists
    (i.e. the phase was actually run), so skipped phases stay absent.
    Safe to call unconditionally — no-op when results.json is missing.
    """
    results_path = Path(output_root) / method / scene / "results.json"
    if not results_path.exists():
        return
    try:
        with open(results_path) as f:
            data = json.load(f)
        tt = _load_training_time(method, scene, output_root)
        rt = _load_render_time(method, scene, output_root)
        if tt:
            data["training_time_s"] = tt["seconds"]
            data["training_time"]   = tt["formatted"]
        if rt:
            data["render_time_s"] = rt["seconds"]
            data["render_time"]   = rt["formatted"]
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
    except (json.JSONDecodeError, OSError, KeyError):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_method(method: str, scene: str, data_root: str, output_root: str):
    cfg      = METHOD_CONFIGS[method]
    source   = os.path.join(data_root, scene)
    out_dir  = os.path.join(output_root, method, scene)
    expname  = f"output_baselines/{method}/{scene}"

    cmd = (
        f"python train.py "
        f"-s {source} "
        f"-m {out_dir} "
        f"--expname {expname} "
        f"{cfg['train_args']}"
    )
    print(f"\n[train] {method}/{scene}")
    print(f"  {cmd}")
    t0  = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - t0
    # Only record time when training actually completed (avoids misleading
    # timings from crashed or OOM runs)
    if ret == 0 or is_training_done(method, scene, output_root):
        _save_training_time(method, scene, output_root, elapsed)
    if ret != 0:
        print(f"  [WARNING] Training returned non-zero exit code {ret}")
    print(f"  [time] training took {_fmt_time(elapsed)}")


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_method(method: str, scene: str, output_root: str, iteration: int = -1):
    cfg      = METHOD_CONFIGS[method]
    out_dir  = os.path.join(output_root, method, scene)
    iter_arg = f"--iteration {iteration}" if iteration > 0 else ""

    # Render test split only (metrics.py evaluates test split)
    cmd = (
        f"python render.py "
        f"-m {out_dir} "
        f"{iter_arg} "
        f"--model_class {cfg['model_class']} "
        f"--skip_train "
        f"--skip_video "
        f"--quiet"
    )
    print(f"\n[render] {method}/{scene}")
    print(f"  {cmd}")
    t0      = time.time()
    ret     = os.system(cmd)
    elapsed = time.time() - t0
    # Only record time when rendering actually completed
    if ret == 0 or is_rendering_done(method, scene, output_root):
        _save_render_time(method, scene, output_root, elapsed)
    if ret != 0:
        print(f"  [WARNING] Rendering returned non-zero exit code {ret}")
    print(f"  [time] rendering took {_fmt_time(elapsed)}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_metrics(method: str, scene: str, output_root: str, eval_flow: bool = False):
    out_dir  = os.path.join(output_root, method, scene)
    flow_arg = "--eval_flow" if eval_flow else ""

    cmd = f"python metrics.py -m {out_dir} {flow_arg}"
    print(f"\n[metrics] {method}/{scene}")
    print(f"  {cmd}")
    os.system(cmd)


def sample_method(method: str, scene: str, output_root: str):
    """
    Extract the first rendered frame per camera pose into <output_root>/<method>/<scene>/sample/.

    Reads cameras_metadata.json from the most recent test/ours_N/renders/ directory
    and copies pose_1.png … pose_N.png.  Writes a .done sentinel on success so
    full_eval.py can resume correctly after an interruption.
    """
    test_dir = Path(output_root) / method / scene / "test"
    if not test_dir.is_dir():
        print(f"  [WARNING] {method}/{scene}: no test/ dir — run rendering first.")
        return

    # Find the latest ours_N directory
    ours_dirs = sorted(
        [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("ours_")],
        key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else 0,
    )
    if not ours_dirs:
        print(f"  [WARNING] {method}/{scene}: no ours_* dir found in test/ — skipping samples.")
        return

    renders_dir = ours_dirs[-1] / "renders"
    meta_path   = renders_dir / "cameras_metadata.json"
    if not meta_path.exists():
        print(f"  [WARNING] {method}/{scene}: cameras_metadata.json missing — skipping samples.")
        return

    with open(meta_path) as f:
        cam_meta = json.load(f)  # {"00000.png": "cam_name", ...}

    # Group filenames by camera, sort within each camera
    cam_frames: dict = {}
    for fname, cam_name in cam_meta.items():
        cam_frames.setdefault(cam_name, []).append(fname)

    if not cam_frames:
        print(f"  [WARNING] {method}/{scene}: cameras_metadata.json is empty.")
        return

    sample_dir = Path(output_root) / method / scene / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sorted_cams = sorted(cam_frames.keys())
    count = 0
    for pose_idx, cam_name in enumerate(sorted_cams, start=1):
        first_frame = sorted(cam_frames[cam_name])[0]  # earliest timestep
        src = renders_dir / first_frame
        dst = sample_dir / f"pose_{pose_idx}.png"
        if src.exists():
            shutil.copy2(str(src), str(dst))
            count += 1

    if count > 0:
        # Write sentinel so is_samples_done() is unambiguous
        (sample_dir / ".done").write_text(f"{count} poses\n")
        print(f"\n[sample] {method}/{scene}  →  {count} pose images in {sample_dir}")
    else:
        print(f"  [WARNING] {method}/{scene}: no pose images could be copied.")


# ─────────────────────────────────────────────────────────────────────────────
# Results loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(method: str, scene: str, output_root: str) -> dict:
    """
    Load results.json and return {"test": {metric: value}, "train": {metric: value}}.
    Keys in results.json prefixed with "train_" go to the train bucket; others to test.
    Missing file or split → empty dict for that split.
    """
    results_path = Path(output_root) / method / scene / "results.json"
    if not results_path.exists():
        return {"test": {}, "train": {}}
    with open(results_path) as f:
        data = json.load(f)
    test_acc:  dict = {}
    train_acc: dict = {}
    for key, sub in data.items():
        if not isinstance(sub, dict):
            continue
        bucket = train_acc if key.startswith("train_") else test_acc
        for metric, v in sub.items():
            bucket.setdefault(metric, []).append(v)
    tt = _load_training_time(method, scene, output_root)
    rt = _load_render_time(method, scene, output_root)
    return {
        "test":  {k: float(np.mean(v)) for k, v in test_acc.items()},
        "train": {k: float(np.mean(v)) for k, v in train_acc.items()},
        "training_time":   tt.get("formatted", "N/A"),
        "training_time_s": tt.get("seconds"),
        "render_time":     rt.get("formatted", "N/A"),
        "render_time_s":   rt.get("seconds"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Merge all method results into output_baselines/result.json
# ─────────────────────────────────────────────────────────────────────────────

def merge_results(methods: list, scenes: list, output_root: str) -> dict:
    """
    Collect per-method results.json files and write a merged result.json.

    Structure:
        {
          "scene": {
            "method": {
              "test":  {"PSNR": 28.3, "SSIM": 0.88, ...},
              "train": {"PSNR": 31.2, "SSIM": 0.92, ...}
            }, ...
          }, ...
        }
    """
    merged: dict = {}
    for scene in scenes:
        merged[scene] = {}
        for method in methods:
            merged[scene][method] = load_results(method, scene, output_root)

    out_path = Path(output_root) / "result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n[results] Written to {out_path}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Console comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(methods: list, scenes: list, merged: dict,
                           metrics=None):
    if metrics is None:
        metrics = METRICS
    col_w  = 11
    time_w = 12
    lbl_w  = 28  # method + split label width
    width  = lbl_w + col_w * len(metrics) + time_w * 2

    for scene in scenes:
        print("\n" + "=" * width)
        print(f"BASELINE COMPARISON  —  {scene}")
        print("=" * width)
        header = (
            f"  {'Method / Split':<{lbl_w-2}}"
            + "".join(f"  {m:>{col_w-2}}" for m in metrics)
            + f"  {'Train Time':>{time_w-2}}"
            + f"  {'Render Time':>{time_w-2}}"
        )
        print(header)
        for method in methods:
            print("  " + "-" * (lbl_w - 2 + col_w * len(metrics) + time_w * 2))
            tt = merged.get(scene, {}).get(method, {}).get("training_time", "N/A")
            rt = merged.get(scene, {}).get(method, {}).get("render_time",   "N/A")
            for split in ["test", "train"]:
                res   = merged.get(scene, {}).get(method, {}).get(split, {})
                label = f"{method} [{split}]"
                row   = f"  {label:<{lbl_w-2}}"
                for m in metrics:
                    v = res.get(m)
                    row += f"  {v:>{col_w-2}.4f}" if v is not None else f"  {'N/A':>{col_w-2}}"
                # Train time belongs to the train split; render time to the test split
                row += f"  {(tt if split == 'train' else ''):>{time_w-2}}"
                row += f"  {(rt if split == 'test'  else ''):>{time_w-2}}"
                print(row)

    print("\n" + "=" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table image (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_table_image(methods: list, scenes: list, merged: dict,
                                output_root: str, metrics=None):
    """
    Render a matplotlib table image comparing all methods across scenes.
    Saved to <output_root>/comparison_table.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
    except ImportError:
        print("[viz] matplotlib not available — skipping comparison table image.")
        return

    if metrics is None:
        metrics = METRICS
    display_metrics = list(metrics) + ["Train Time", "Render Time"]

    # Build rows: two rows per method (test + train) per scene
    row_labels  = []
    cell_values = []
    row_meta    = []  # (scene_idx, split) for colouring

    for s_idx, scene in enumerate(scenes):
        for method in methods:
            tt = merged.get(scene, {}).get(method, {}).get("training_time", "N/A")
            rt = merged.get(scene, {}).get(method, {}).get("render_time",   "N/A")
            for split in ["test", "train"]:
                res = merged.get(scene, {}).get(method, {}).get(split, {})
                row_labels.append(f"{method}\n[{split}]")
                # Train time belongs to the train split; render time to the test split
                tt_cell = tt if split == "train" else ""
                rt_cell = rt if split == "test"  else ""
                cell_values.append([
                    f"{res[m]:.4f}" if res.get(m) is not None else "N/A"
                    for m in metrics
                ] + [tt_cell, rt_cell])
                row_meta.append((s_idx, split))

    n_rows = len(row_labels)
    n_cols = len(display_metrics)

    fig_height = max(4, n_rows * 0.45 + 1.5)
    fig_width  = max(10, n_cols * 1.8 + 3.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_values,
        rowLabels=row_labels,
        colLabels=display_metrics,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)

    # Colour header row
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#2c4770")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Colour rows: alternating per method-block; test slightly darker than train
    scene_base = ["#dce8ff", "#ffeedd"]
    for r_idx, (s_idx, split) in enumerate(row_meta):
        base = scene_base[s_idx % len(scene_base)]
        if split == "train":
            # lighten the base colour slightly
            import colorsys
            r, g, b = int(base[1:3],16)/255, int(base[3:5],16)/255, int(base[5:7],16)/255
            h, s_v, v = colorsys.rgb_to_hsv(r, g, b)
            r2, g2, b2 = colorsys.hsv_to_rgb(h, max(0, s_v - 0.15), min(1, v + 0.06))
            bg = "#{:02x}{:02x}{:02x}".format(int(r2*255), int(g2*255), int(b2*255))
        else:
            bg = base
        row_tbl = r_idx + 1  # +1 for header
        for col in range(n_cols):
            tbl[row_tbl, col].set_facecolor(bg)

    fig.suptitle("Baseline Comparison — coffee_martini", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_path = Path(output_root) / "comparison_table.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Comparison table saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Baseline comparison for ODE 4D Gaussian Splatting.",
        epilog=(
            "Single-method examples:\n"
            "  python full_eval.py --method ode\n"
            "  python full_eval.py --method fourier_approx --skip_training\n"
            "  python full_eval.py --method neural_ode --skip_training --skip_rendering\n"
            "\nFull comparison:\n"
            "  python full_eval.py                    # all methods\n"
            "  python full_eval.py --skip_training    # render + metrics only"
        ),
    )
    parser.add_argument("--data_root",        type=str, default=DEFAULT_DATA_ROOT,
                        help="Root containing scene sub-folders (default: data/multipleview).")
    parser.add_argument("--output_root",      type=str, default=DEFAULT_OUTPUT,
                        help="Root output directory (default: output/output_baselines).")
    parser.add_argument("--scenes",           nargs="+", type=str, default=[DEFAULT_SCENE],
                        help="Scenes to evaluate (default: coffee_martini).")
    # ── Method selection: --method (single) OR --methods (subset/all) ────────
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--method", type=str, default=None, choices=ALL_METHODS,
        help="Run a single method end-to-end (train → render → metrics).",
    )
    method_group.add_argument(
        "--methods", nargs="+", type=str, default=ALL_METHODS, choices=ALL_METHODS,
        help="Run an explicit subset of methods (default: all).",
    )
    parser.add_argument("--render_iteration", type=int,  default=-1,
                        help="Iteration to render (-1 = latest checkpoint).")
    parser.add_argument("--skip_training",    action="store_true")
    parser.add_argument("--skip_rendering",   action="store_true")
    parser.add_argument("--skip_metrics",     action="store_true")
    parser.add_argument("--eval_flow",        action="store_true",
                        help="Include optical-flow EPE in metrics (requires RAFT).")
    args = parser.parse_args()

    # Resolve --method (single shorthand) → args.methods
    if args.method is not None:
        args.methods = [args.method]

    # ── Hardware info ─────────────────────────────────────────────────────────
    log_gpu_info()
    if _GPU_MEM_FLAGS:
        print(f"[hardware] densify threshold flags: {_GPU_MEM_FLAGS.strip()}")
    if GPU_CFG.cli_num_workers_flag:
        print(f"[hardware] DataLoader workers override: {GPU_CFG.cli_num_workers_flag.strip()}")
    print(f"[hardware] Methods to run: {args.methods}")

    # ── Training ──────────────────────────────────────────────────────────
    if not args.skip_training:
        for method in args.methods:
            for scene in args.scenes:
                if is_training_done(method, scene, args.output_root):
                    print(f"\n[train] SKIP {method}/{scene}  (final checkpoint already exists)")
                else:
                    train_method(method, scene, args.data_root, args.output_root)

    # ── Rendering ─────────────────────────────────────────────────────────
    if not args.skip_rendering:
        for method in args.methods:
            for scene in args.scenes:
                if is_rendering_done(method, scene, args.output_root):
                    print(f"\n[render] SKIP {method}/{scene}  (rendered images already exist)")
                else:
                    render_method(method, scene, args.output_root, args.render_iteration)
    # ── Sample extraction (first rendered frame per camera pose) ──────────────
    # Runs unconditionally (not gated by --skip_rendering) so that a run
    # interrupted between rendering and sample extraction resumes cleanly.
    for method in args.methods:
        for scene in args.scenes:
            if is_samples_done(method, scene, args.output_root):
                print(f"\n[sample] SKIP {method}/{scene}  (samples already exist)")
            else:
                sample_method(method, scene, args.output_root)
    # ── Metrics ───────────────────────────────────────────────────────────
    if not args.skip_metrics:
        for method in args.methods:
            for scene in args.scenes:
                if is_metrics_done(method, scene, args.output_root):
                    print(f"\n[metrics] SKIP {method}/{scene}  (results.json already exists)")
                else:
                    run_metrics(method, scene, args.output_root, eval_flow=args.eval_flow)

    # Always patch training_time into each per-method results.json (no-op when
    # either file is absent).  Runs outside the skip_metrics gate so the time
    # is recorded even when metrics were skipped or computed in a prior run.
    for method in args.methods:
        for scene in args.scenes:
            _patch_results_json(method, scene, args.output_root)

    # ── Merge + save result.json ──────────────────────────────────────────
    merged = merge_results(args.methods, args.scenes, args.output_root)

    # ── Console comparison table ──────────────────────────────────────────
    print_comparison_table(args.methods, args.scenes, merged)

    # ── Comparison table image ─────────────────────────────────────────────
    save_comparison_table_image(args.methods, args.scenes, merged, args.output_root)

