# Dynamic 3D Gaussian Splatting with Explicit ODE-Based Mean and Covariance Evolution

This repository extends [4D Gaussian Splatting (Wu et al., CVPR 2024)](https://guanjunwu.github.io/4dgs/index.html) by replacing the HexPlane + MLP deformation network with **explicit per-Gaussian ODE parameters**.  The rendering backbone (standard 3DGS rasterizer) is completely unchanged.  The only modification is how Gaussian mean and covariance evolve over time.

---

## Overview

| Component | Original 4DGaussians | This work |
|---|---|---|
| Temporal model | HexPlane + MLP | Per-Gaussian ODE parameters |
| Mean evolution | MLP-predicted Δxyz per frame | Closed-form ODE solution at τ |
| Covariance evolution | MLP-predicted Δscale, Δrotation | Explicit linear ODEs |
| Parameters | ~50 MB HexPlane grid + MLP | ~11 scalars per Gaussian |
| Training script | `train.py` | `train.py` (ODE only) |

### Two ODE variants for mean evolution

**Complex-valued ODE** (default, `--ode_type complex`):
```
dz_i/dτ = a_i · z_i + b_i,   a_i, b_i ∈ ℂ,   z_i(0) = z_i^0
μ_i(τ) = μ_i^0 + Re(z_i(τ)) u_i + Im(z_i(τ)) v_i
```
Compactly represents circular, spiral, oscillatory, and drifting planar trajectories.

**Real-valued ODE** (`--ode_type real`):
```
dp_i/dτ = A_i p_i + d_i,   A_i ∈ ℝ^{2×2},   p_i(0) = p_i^0
μ_i(τ) = μ_i^0 + p_{i1}(τ) u_i + p_{i2}(τ) v_i
```

**Covariance ODEs** (same for both variants):
```
ds_{ij}/dτ = κ_{ij}     →  log-scale evolves linearly
dθ_i/dτ    = ω_i        →  in-plane orientation evolves linearly
```

Time is normalised: `τ = 2t − 1` for `t ∈ [0, 1]`.

---

## Installation

Follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) setup instructions, then:

```bash
git clone <this-repo>
cd 4DGaussians
git submodule update --init --recursive

conda create -n Gaussians4D python=3.7
conda activate Gaussians4D
pip install -r requirements.txt

# Build CUDA extensions (requires CUDA 11.7 and gcc-11)
export CUDA_HOME=/usr/local/cuda-11.7
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

---

## Data Preparation

**D-NeRF (synthetic):** Download from [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0) and place under `data/dnerf/`.

**HyperNeRF (real):** Download from [HyperNeRF Dataset](https://github.com/google/hypernerf/releases/tag/v0.1), place under `data/hypernerf/`.

**Neu3D / DyNeRF (multi-camera):**
```bash
# Extract frames
python scripts/preprocess_dynerf.py --datadir data/dynerf/<scene>
# Generate point cloud
bash colmap.sh data/dynerf/<scene> llff
# Downsample
python scripts/downsample_point.py \
    data/dynerf/<scene>/colmap/dense/workspace/fused.ply \
    data/dynerf/<scene>/points3D_downsample2.ply
```

**Custom multi-view data:**
```bash
# Organise frames as data/multipleview/<name>/cam01/frame_XXXXX.jpg ...
bash multipleviewprogress.sh <name>
```

Expected folder structure:
```
data/
  dnerf/        bouncingballs/  hellwarrior/  ...
  hypernerf/    virg/  interp/  ...
  dynerf/       coffee_martini/  cut_roasted_beef/  ...
  multipleview/ <name>/
```

---

## Training

### ODE model (this work)

```bash
# Complex ODE on D-NeRF bouncingballs (recommended)
python train.py \
    -s data/dnerf/bouncingballs \
    --ode_type complex \
    --lambda_ode 0.001 \
    --expname "ode_complex/bouncingballs" \
    --configs arguments/dnerf/bouncingballs.py

# Real ODE variant
python train.py \
    -s data/dnerf/bouncingballs \
    --ode_type real \
    --expname "ode_real/bouncingballs" \
    --configs arguments/dnerf/bouncingballs.py

# DyNeRF scene
python train.py \
    -s data/dynerf/cut_roasted_beef \
    --ode_type complex \
    --expname "ode_complex/cut_roasted_beef" \
    --configs arguments/dynerf/cut_roasted_beef.py

# HyperNeRF scene
python train.py \
    -s data/hypernerf/virg/broom2 \
    --ode_type complex \
    --expname "ode_complex/broom2" \
    --configs arguments/hypernerf/broom2.py
```

### With checkpoint

```bash
# Save checkpoints every 5000 iterations
python train.py -s data/dnerf/bouncingballs \
    --expname "ode_complex/bouncingballs" \
    --configs arguments/dnerf/bouncingballs.py \
    --checkpoint_iterations 5000 10000

# Resume from checkpoint
python train.py -s data/dnerf/bouncingballs \
    --expname "ode_complex/bouncingballs" \
    --configs arguments/dnerf/bouncingballs.py \
    --start_checkpoint output/ode_complex/bouncingballs/chkpnt_coarse_5000.pth
```

### Key training flags

| Flag | Description | Default |
|---|---|---|
| `--ode_type` | `complex` or `real` | `complex` |
| `--lambda_ode` | ODE regularisation weight | `0.001` |
| `--lambda_flow` | Optical flow consistency weight (0=off) | `0.0` |
| `--lambda_dssim` | SSIM loss weight | `0.2` |
| `--configs` | Scene-specific config file | — |
| `--expname` | Experiment name (sets output path) | — |

---

## Rendering

```bash
python render.py \
    --model_path output/ode_complex/bouncingballs \
    --skip_train \
    --configs arguments/dnerf/bouncingballs.py
```

---

## Evaluation

### Rendering metrics (PSNR, SSIM, LPIPS, MS-SSIM)

```bash
python metrics.py -m output/ode_complex/bouncingballs
```

### With trajectory visualisation (ODE models)

```bash
python metrics.py \
    -m output/ode_complex/bouncingballs \
    --analyze_trajectories \
    --ode_type complex
```

This saves to `output/ode_complex/bouncingballs/visualizations/`:
- `trajectory/trajectory_curves.png` — 3D trajectory curves of the most dynamic Gaussians
- `trajectory/velocity_profile.png` — mean speed vs normalised time τ
- `trajectory/acceleration_profile.png` — mean acceleration magnitude vs τ
- `trajectory/jitter_heatmap.png` — XY scatter coloured by per-Gaussian jitter score

### With optical flow error

```bash
python metrics.py \
    -m output/ode_complex/bouncingballs \
    --eval_flow
```
Requires torchvision ≥ 0.15 (RAFT optical flow model).

---

## Baseline Comparison

Run all five methods on a dataset and produce a comparison table + plots:

```bash
python full_eval.py \
    --data_root data/dnerf \
    --output_path output/comparison \
    --dataset dnerf \
    --scenes bouncingballs jumpingjacks lego
```

Methods compared:
1. **ode_complex** — complex ODE (proposed, `--ode_type complex`)
2. **ode_real** — real ODE (`--ode_type real`)
3. **linear** — linear approximation (ODE with frozen dynamics matrix A=0)

Outputs:
- Console comparison table (PSNR / SSIM / LPIPS / MS-SSIM)
- `output/comparison/comparison_plots/<scene>_PSNR.png` and similar bar charts
- `output/comparison/comparison_plots/mean_psnr_summary.png`
- Per-method `results.json` and trajectory visualisations

```bash
# Skip training (already done) and just re-run metrics + table
python full_eval.py \
    --data_root data/dnerf \
    --output_path output/comparison \
    --dataset dnerf \
    --skip_training --skip_rendering

# With flow error and trajectory analysis
python full_eval.py \
    --data_root data/dnerf \
    --output_path output/comparison \
    --dataset dnerf \
    --skip_training --skip_rendering \
    --eval_flow --analyze_trajectories
```

---

## Visualisation (Interactive Viewer)

The 3DGS network GUI works with the ODE model.  Launch the viewer and connect:

```bash
python train.py -s data/dnerf/bouncingballs --port 6017 \
    --expname "ode_complex/bouncingballs" \
    --configs arguments/dnerf/bouncingballs.py
# Open SIBR viewer and connect to localhost:6017
```

See [docs/viewer_usage.md](docs/viewer_usage.md) for full viewer setup instructions.

---

## Repository Structure

```
4DGaussians/
├── train.py                        ODE training script
├── render.py                       Rendering
├── metrics.py                      Evaluation + ODE trajectory visualisation
├── full_eval.py                    Multi-method baseline comparison
├── scene/
│   ├── ode_deformation.py          ODE math (complex/real ODE solvers, covariance)
│   └── gaussian_model_ode.py       GaussianModelODE class
├── gaussian_renderer/
│   └── __init__.py                 render() and render_ode() functions
├── arguments/
│   └── __init__.py                 All arg groups incl. ODEModelParams, ODEOptimizationParams
└── utils/
    └── loss_utils.py               Photometric + SSIM + LPIPS losses
```

---

## Notes

- **Densification** works identically to the original: ODE parameters are per-Gaussian and are zero-initialised for newly split/cloned Gaussians.
- **`scene/__init__.py` compatibility**: the ODE model uses a `_NoOpDeformation` shim so that `scene/__init__.py`'s call to `set_aabb()` does not crash.
- **Flow loss**: set `--lambda_flow > 0` only if you have access to an optical flow estimator (RAFT); by default it is disabled.
- **Linear baseline**: equivalent to `--ode_type real` with the dynamics matrix `A` frozen to zero.  In `full_eval.py` this is approximated by passing `--ode_lr_init 0.0` for the A_flat parameter group.

---

## Citation

If you use this work, please also cite the original 4DGaussians paper:

```bibtex
@InProceedings{Wu_2024_CVPR,
    author    = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and
                 Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
    title     = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
    booktitle = {CVPR},
    year      = {2024},
}
```

Some source code is borrowed from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [K-Planes](https://github.com/Giodiro/kplanes_nerfstudio), [HexPlane](https://github.com/Caoang327/HexPlane), [TiNeuVox](https://github.com/hustvl/TiNeuVox), and [Depth-Rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization).


