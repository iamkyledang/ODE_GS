#
# full_eval.py  —  Baseline comparison for ODE-based dynamic 4D Gaussian Splatting.
#
# Trains five temporal-deformation models on coffee_martini (MultipleView dataset)
# and evaluates them with the full metrics suite.  All method configs are declared
# inline below — no external config files are needed.
#
# Methods compared:
#   1. ode                    — proposed: 3D ODE deformation (this repo)
#   2. deformable_mlp         — per-Gaussian deformation MLP (ingra14m/Deformable-3D-Gaussians)
#   3. deformable_hexplane_mlp — HexPlane + MLP (hustvl/4DGaussians)
#   4. fourier_approx         — per-Gaussian Fourier series approximation
#   5. polynomial_approx      — per-Gaussian polynomial approximation
#
# Usage:
#   python full_eval.py                            # train + render + metrics for all methods
#   python full_eval.py --skip_training            # skip training (use existing checkpoints)
#   python full_eval.py --skip_rendering           # skip rendering
#   python full_eval.py --skip_metrics             # skip metrics computation
#   python full_eval.py --methods ode fourier_approx  # run a subset of methods
#   python full_eval.py --data_root /path/to/data  # custom data location
#
# All outputs are written under output/output_baselines/ (relative to cwd).
#

import os
import json
import numpy as np
from argparse import ArgumentParser
from pathlib import Path


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

# Shared base flags for MultipleView / coffee_martini
_BASE_FLAGS = (
    "--dataloader "
    "--batch_size 1 "
    "--opacity_threshold_coarse 0.005 "
    "--opacity_threshold_fine_init 0.005 "
    "--opacity_threshold_fine_after 0.005 "
    "--quiet"
)

METHOD_CONFIGS = {
    # ── Proposed: ODE deformation ──────────────────────────────────────────
    "ode": dict(
        model_class="ode",
        train_args=(
            "--model_class ode "
            "--iterations 100000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 15000 "
            "--lambda_ode 0.001 "
            "--lambda_dssim 0.2 "
            "--net_width 64 "          # ODE hidden-network width
            "--bounds 1.6 "
            "--plane_tv_weight 0.0 "
            + _BASE_FLAGS
        ),
    ),

    # ── Deformable 3D Gaussians (MLP) ────────────────────────────────────
    # Based on: ingra14m/Deformable-3D-Gaussians (CVPR 2024)
    # Arch params (all editable here):
    #   deform_mlp_width  : hidden layer width        (default 256)
    #   deform_mlp_depth  : number of hidden layers   (default 8)
    #   deform_pos_pe     : pos encoding frequencies  (default 10 → pos_ch=63)
    #   deform_time_pe    : time encoding frequencies (default 4  → t_ch=9)
    "deformable_mlp": dict(
        model_class="deformable_mlp",
        train_args=(
            "--model_class deformable_mlp "
            "--iterations 100000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 15000 "
            "--lambda_ode 0.0 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--plane_tv_weight 0.0 "
            "--deform_mlp_width 256 "
            "--deform_mlp_depth 8 "
            "--deform_pos_pe 10 "
            "--deform_time_pe 4 "
            + _BASE_FLAGS
        ),
    ),

    # ── 4D Gaussians (HexPlane + MLP) ────────────────────────────────────
    # Based on: hustvl/4DGaussians (CVPR 2024)
    # Arch params (all editable here):
    #   hexplane_spatial_res : spatial grid resolution x/y/z  (default 64)
    #   hexplane_time_res    : temporal grid resolution t      (default 150)
    #   hexplane_feat_dim    : output feature dim per plane    (default 16)
    #   hexplane_decode_W    : decoder MLP hidden width        (default 128)
    #   hexplane_decode_D    : decoder MLP hidden layers       (default 1)
    "deformable_hexplane_mlp": dict(
        model_class="deformable_hexplane_mlp",
        train_args=(
            "--model_class deformable_hexplane_mlp "
            "--iterations 100000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 10000 "
            "--lambda_ode 0.0 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--plane_tv_weight 0.0002 "
            "--hexplane_spatial_res 64 "
            "--hexplane_time_res 150 "
            "--hexplane_feat_dim 16 "
            "--hexplane_decode_W 128 "
            "--hexplane_decode_D 1 "
            + _BASE_FLAGS
        ),
    ),

    # ── Fourier approximation ────────────────────────────────────────────
    # Based on: raven38/EfficientDynamic3DGaussian
    # Arch param (editable here):
    #   fourier_K : number of sin/cos frequency components (default 4)
    "fourier_approx": dict(
        model_class="fourier_approx",
        train_args=(
            "--model_class fourier_approx "
            "--iterations 100000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 15000 "
            "--lambda_ode 0.0 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--plane_tv_weight 0.0 "
            "--fourier_K 4 "
            + _BASE_FLAGS
        ),
    ),

    # ── Polynomial approximation ─────────────────────────────────────────
    # Arch param (editable here):
    #   poly_D : polynomial degree (fits tau^1 … tau^D, no constant term) (default 4)
    "polynomial_approx": dict(
        model_class="polynomial_approx",
        train_args=(
            "--model_class polynomial_approx "
            "--iterations 100000 "
            "--coarse_iterations 3000 "
            "--densify_until_iter 15000 "
            "--lambda_ode 0.0 "
            "--lambda_dssim 0.2 "
            "--bounds 1.6 "
            "--plane_tv_weight 0.0 "
            "--poly_D 4 "
            + _BASE_FLAGS
        ),
    ),
}

ALL_METHODS = list(METHOD_CONFIGS.keys())
METRICS     = ["PSNR", "SSIM", "LPIPS-vgg", "LPIPS-alex", "MS-SSIM", "D-SSIM"]


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
    ret = os.system(cmd)
    if ret != 0:
        print(f"  [WARNING] Training returned non-zero exit code {ret}")


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
    ret = os.system(cmd)
    if ret != 0:
        print(f"  [WARNING] Rendering returned non-zero exit code {ret}")


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
    return {
        "test":  {k: float(np.mean(v)) for k, v in test_acc.items()},
        "train": {k: float(np.mean(v)) for k, v in train_acc.items()},
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
    lbl_w  = 28  # method + split label width
    width  = lbl_w + col_w * len(metrics)

    for scene in scenes:
        print("\n" + "=" * width)
        print(f"BASELINE COMPARISON  —  {scene}")
        print("=" * width)
        header = f"  {'Method / Split':<{lbl_w-2}}" + "".join(f"  {m:>{col_w-2}}" for m in metrics)
        print(header)
        for method in methods:
            print("  " + "-" * (lbl_w - 2 + col_w * len(metrics)))
            for split in ["test", "train"]:
                res   = merged.get(scene, {}).get(method, {}).get(split, {})
                label = f"{method} [{split}]"
                row   = f"  {label:<{lbl_w-2}}"
                for m in metrics:
                    v = res.get(m)
                    row += f"  {v:>{col_w-2}.4f}" if v is not None else f"  {'N/A':>{col_w-2}}"
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

    # Build rows: two rows per method (test + train) per scene
    row_labels  = []
    cell_values = []
    row_meta    = []  # (scene_idx, split) for colouring

    for s_idx, scene in enumerate(scenes):
        for method in methods:
            for split in ["test", "train"]:
                res = merged.get(scene, {}).get(method, {}).get(split, {})
                row_labels.append(f"{method}\n[{split}]")
                cell_values.append([
                    f"{res[m]:.4f}" if res.get(m) is not None else "N/A"
                    for m in metrics
                ])
                row_meta.append((s_idx, split))

    n_rows = len(row_labels)
    n_cols = len(metrics)

    fig_height = max(4, n_rows * 0.45 + 1.5)
    fig_width  = max(10, n_cols * 1.8 + 3.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_values,
        rowLabels=row_labels,
        colLabels=metrics,
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
    split_shade = {"test": 0, "train": 40}   # lighten train row
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
    parser = ArgumentParser(description="Baseline comparison for ODE 4D Gaussian Splatting")
    parser.add_argument("--data_root",        type=str, default=DEFAULT_DATA_ROOT,
                        help="Root containing scene sub-folders (default: data/multipleview).")
    parser.add_argument("--output_root",      type=str, default=DEFAULT_OUTPUT,
                        help="Root output directory (default: output/output_baselines).")
    parser.add_argument("--scenes",           nargs="+", type=str, default=[DEFAULT_SCENE],
                        help="Scenes to evaluate (default: coffee_martini).")
    parser.add_argument("--methods",          nargs="+", type=str, default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help="Methods to run (default: all five).")
    parser.add_argument("--render_iteration", type=int,  default=-1,
                        help="Iteration to render (-1 = latest checkpoint).")
    parser.add_argument("--skip_training",    action="store_true")
    parser.add_argument("--skip_rendering",   action="store_true")
    parser.add_argument("--skip_metrics",     action="store_true")
    parser.add_argument("--eval_flow",        action="store_true",
                        help="Include optical-flow EPE in metrics (requires RAFT).")
    args = parser.parse_args()

    # ── Training ──────────────────────────────────────────────────────────
    if not args.skip_training:
        for method in args.methods:
            for scene in args.scenes:
                train_method(method, scene, args.data_root, args.output_root)

    # ── Rendering ─────────────────────────────────────────────────────────
    if not args.skip_rendering:
        for method in args.methods:
            for scene in args.scenes:
                render_method(method, scene, args.output_root, args.render_iteration)

    # ── Metrics ───────────────────────────────────────────────────────────
    if not args.skip_metrics:
        for method in args.methods:
            for scene in args.scenes:
                run_metrics(method, scene, args.output_root, eval_flow=args.eval_flow)

    # ── Merge + save result.json ──────────────────────────────────────────
    merged = merge_results(args.methods, args.scenes, args.output_root)

    # ── Console comparison table ──────────────────────────────────────────
    print_comparison_table(args.methods, args.scenes, merged)

    # ── Comparison table image ─────────────────────────────────────────────
    save_comparison_table_image(args.methods, args.scenes, merged, args.output_root)

