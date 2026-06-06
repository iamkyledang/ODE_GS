# Dynamic 4D Gaussian Splatting with ODE-Based Deformation

A research codebase for dynamic 3D Gaussian Splatting in which Gaussian mean and covariance evolve through **explicit, closed-form ordinary differential equations** — no neural deformation field is introduced into the renderer. The original 3DGS rasterizer is preserved exactly; only the temporal parameterisation of each Gaussian primitive changes.

Designed as a reproducible comparison framework: the proposed ODE-GS method and five published baselines are trained and evaluated under identical conditions on the Neural 3D Video dataset.

---

## Method overview

Each Gaussian primitive carries per-primitive ODE parameters instead of relying on a shared deformation network:

| Parameter | Shape | Role |
|---|---|---|
| `A` | (N, 3, 3) | Dynamics matrix — controls coupled 3D motion |
| `b` | (N, 3) | Drift vector — constant forcing |
| `x₀` | (N, 3) | Initial displacement at τ = 0 |
| `κ` | (N, 3) | Log-scale rate — covariance size evolution |
| `ω` | (N, 3) | Angular velocity — covariance orientation (SO(3)) |

**Mean trajectory** — closed-form via augmented 4×4 matrix exponential:

```
dx_i/dτ = A_i x_i + b_i,   x_i(0) = x₀_i
μ_i(τ)  = μ_i⁰ + x_i(τ)
```

**Covariance** — rotation–scale decomposition, both evolved analytically:

```
s_j(τ) = s_j⁰ + κ_j τ                    (log-scale, linear ODE)
R(τ)   = exp(ω̂ τ) R⁰                     (SO(3) rotation ODE)
Σ(τ)   = R(τ) diag(e^s(τ)) R(τ)ᵀ
```

**Training loss:**

```
L = L_rec  +  λ_f L_flow  +  λ_r L_ode

L_rec  = Σ_t ‖Î_t − I_t‖₁  +  λ_s (1 − SSIM(Î_t, I_t))
L_flow = Σ_t ‖F̂_{t→t+1} − F_{t→t+1}‖₁     (optional optical-flow consistency)
L_ode  = L_traj  +  λ_ω L_ω  +  λ_s L_s    (trajectory / shape regularisation)
```

---

## Baselines included

| Key | Description | Reference |
|---|---|---|
| `ode` | **Proposed — ODE-GS** (this work) | — |
| `deformable_mlp` | Per-Gaussian displacement MLP | Ingra14m/Deformable-3D-Gaussians (CVPR 2024) |
| `deformable_hexplane_mlp` | HexPlane space–time grid + MLP decoder | hustvl/4DGaussians (CVPR 2024) |
| `fourier_approx` | Per-Gaussian Fourier series (K=2) | raven38/EfficientDynamic3DGaussian |
| `polynomial_approx` | Per-Gaussian polynomial (degree 3) | NJU-3DV/Gaussian-Flow |
| `neural_ode` | Shared velocity MLP + Euler ODE integration | arnold-caleb/evogs |

All reference hyperparameters are sourced directly from each method's published repository config and set in `full_eval.py`.

---

## Hardware requirements

| GPU | VRAM | Behaviour |
|---|---|---|
| RTX 4090 | 24 GB | TF32 matmul, `pin_memory`, batch_size=2 |
| RTX 3060 / other | ≤ 8 GB | FP32, batch_size=1, conservative memory |

Hardware is detected automatically at startup — no manual flags needed.

---

## Installation

```bash
git clone <this-repo> && cd 4DGaussians_ODE3
git submodule update --init --recursive

conda create -n Gaussians4D python=3.8
conda activate Gaussians4D

export CUDA_HOME=/usr/local/cuda-11.7
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

---

## Pipeline

### Step 1 — Download & Preprocess Data

Downloads from the [Neural 3D Video release](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0), extracts frames, and places them in `data/multipleview/<scene>/`.

Available scenes: `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`, `sear_steak`.

```bash
python preprocess_data.py                          # coffee_martini, all frames
python preprocess_data.py --dataset_name cook_spinach
python preprocess_data.py --source_dir /path/to/coffee_martini   # pre-downloaded
python preprocess_data.py --stride 5               # every 5th frame
```

Output layout:
```
data/multipleview/coffee_martini/
  cam00/images/frame_00001.jpg ...
  cam01/images/frame_00001.jpg ...
  poses_bounds.npy
```

> Camera indices are renumbered automatically when gaps are present.

---

### Step 2 — COLMAP & Point Cloud

```bash
bash multipleviewprogress.sh coffee_martini
```

Produces `data/multipleview/coffee_martini/points3D_multipleview.ply`.

---

### Step 3 — Train (single method)

```bash
python train.py \
  -s data/multipleview/coffee_martini \
  -m output/multipleview/coffee_martini \
  --expname multipleview/coffee_martini \
  --model_class ode \
  --iterations 30000 \
  --dataloader
```

Key training flags:

| Flag | Description | Default |
|---|---|---|
| `--model_class` | `ode`, `deformable_mlp`, `deformable_hexplane_mlp`, `fourier_approx`, `polynomial_approx`, `neural_ode` | `ode` |
| `--iterations` | Fine-stage training iterations | `30000` |
| `--coarse_iterations` | Coarse-stage iterations (canonical Gaussians only) | `3000` |
| `--lambda_ode` | ODE regularisation weight | `0.001` |
| `--lambda_dssim` | SSIM loss weight | `0.2` |
| `--ode_lr_init` | Initial ODE parameter learning rate | `0.001` |
| `--scaling_lr` | Canonical scale learning rate | `0.001` |
| `--dataloader` | Use DataLoader with multi-CPU workers | off |

---

### Step 4 — Render

Renders the test split. The test split contains all `(camera, timestep)` pairs not used for training — one image per combination. Metrics are later computed as a mean over all those pairs.

```bash
python render.py \
  -m output/multipleview/coffee_martini \
  --model_class ode \
  --skip_train \
  --skip_video
```

Output: `output/multipleview/coffee_martini/test/ours_<iteration>/renders/` and `gt/`.

---

### Step 5 — Metrics

```bash
python metrics.py -m output/multipleview/coffee_martini
```

Computes **PSNR, SSIM, LPIPS-vgg, LPIPS-alex, MS-SSIM, D-SSIM** over all test frames (all camera poses and all timesteps merged). Optionally also computes optical-flow EPE between consecutive rendered and ground-truth frames:

```bash
python metrics.py -m output/multipleview/coffee_martini --eval_flow
```

Outputs:
- `results.json` — scalar summary metrics
- `per_view.json` — per-frame metric values
- `visualizations/test/<ours_N>/` — PSNR/SSIM/LPIPS time-series plots; flow error heatmaps (if `--eval_flow`)

---

## Baseline Comparison (`full_eval.py`)

Trains, renders, and evaluates all six methods on one or more scenes. All method hyperparameters are declared inline — no external config files needed.

```bash
# Run everything (all 6 methods, coffee_martini)
python full_eval.py

# Single method end-to-end
python full_eval.py --method ode
python full_eval.py --method fourier_approx
python full_eval.py --method neural_ode

# Explicit subset
python full_eval.py --methods ode fourier_approx polynomial_approx

# Skip phases
python full_eval.py --skip_training                     # render + metrics only
python full_eval.py --method ode --skip_training        # render + metrics for ODE only
python full_eval.py --skip_training --skip_rendering    # metrics + table only

# Different scene
python full_eval.py --scenes cook_spinach

# Custom paths
python full_eval.py --data_root /data/neural3d --output_root /results/baselines
```

Output structure:
```
output/output_baselines/
  result.json              ← merged metrics for all methods and scenes
  comparison_table.png     ← matplotlib comparison image
  ode/coffee_martini/
    cfg_args
    point_cloud/iteration_30000/point_cloud.ply
    deformation/iteration_30000/ode_deformation.pth
    test/ours_30000/renders/*.png
    test/ours_30000/gt/*.png
    results.json
    per_view.json
    visualizations/
  deformable_mlp/coffee_martini/  ...   (iteration_40000)
  deformable_hexplane_mlp/...           (iteration_15000)
  fourier_approx/...                    (iteration_30000)
  polynomial_approx/...                 (iteration_30000)
  neural_ode/...                        (iteration_14000)
```

Resume behaviour: each phase (train / render / metrics) is skipped automatically if its output already exists.

---

## Project structure

```
4DGaussians_ODE3/
  train.py                  — main training script
  render.py                 — rendering script
  metrics.py                — evaluation + visualisation
  full_eval.py              — end-to-end baseline comparison pipeline
  preprocess_data.py        — dataset download + frame extraction
  framework.tex             — mathematical description of the ODE model
  arguments/__init__.py     — ODEModelParams + ODEOptimizationParams
  scene/
    gaussian_model.py       — ODE-GS model (proposed method)
    deformation.py          — closed-form ODE solvers (matrix exp, SO(3))
    dataset_readers.py      — MultipleView / DyNeRF / HyperNeRF loaders
  baselines/
    deformable_MLP.py       — displacement MLP baseline
    deformable_hexplane_MLP.py  — HexPlane + MLP baseline
    fourier_approx.py       — Fourier series baseline
    polynomial_approx.py    — polynomial baseline
    neural_ode.py           — velocity MLP + Euler ODE baseline
  gaussian_renderer/
    __init__.py             — render_ode() — unmodified 3DGS rasterizer wrapper
  reference_repos/          — cloned reference implementations (eval only)
    Deformable-3D-Gaussians/
    4DGaussians/
    EfficientDynamic3DGaussian/
    Gaussian-Flow/
    evogs/
```

---

## Citation

If you use this codebase or the ODE-GS method, please cite the underlying 3DGS and 4DGS works:

```bibtex
@InProceedings{Wu_2024_CVPR,
    author    = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and
                 Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
    title     = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
    booktitle = {CVPR},
    year      = {2024},
}

@article{kerbl3Dgaussians,
    author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal   = {ACM Transactions on Graphics},
    volume    = {42},
    number    = {4},
    year      = {2023},
}
```

