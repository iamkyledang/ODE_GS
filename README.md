# ODE-GS: Dynamic 3D Gaussian Splatting via Closed-Form ODEs

This project proposes **ODE-GS**, a method for rendering dynamic 3D scenes by driving each Gaussian primitive with a closed-form Ordinary Differential Equation (ODE) — eliminating the need for neural deformation networks. It bridges **computer vision**, **differential equations**, and **machine learning** into a unified rendering pipeline.

The method is benchmarked against five established baselines under identical conditions on the [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) dataset (`coffee_martini` scene).

---

## How It Works

Each 3D Gaussian has its own ODE parameters that analytically describe how it moves and deforms over time:

- **Position** evolves via a matrix exponential (linear ODE)
- **Scale** changes linearly in log-space
- **Rotation** follows the SO(3) exponential map

This keeps the 3DGS rasterizer unchanged — only the temporal parameterization is replaced.

**Training loss:** reconstruction + optical flow supervision + ODE regularization

---

## Methods Compared

| Key | Method | Source |
|---|---|---|
| `ode` | **ODE-GS** (this work) | — |
| `deformable_mlp` | Displacement MLP | Deformable-3D-Gaussians (CVPR 2024) |
| `deformable_hexplane_mlp` | HexPlane + MLP | 4DGaussians (CVPR 2024) |
| `fourier_approx` | Fourier series (K=2) | EfficientDynamic3DGaussian |
| `polynomial_approx` | Polynomial (degree 3) | Gaussian-Flow |
| `neural_ode` | Velocity MLP + Euler integration | evogs |

---

## Results

Pre-computed results for all methods on the `coffee_martini` scene are stored in `results/`:

```
results/
├── final_result.json              ← aggregated metrics for all methods
├── ode/coffee_martini/
├── deformable_mlp/coffee_martini/
├── deformable_hexplane_mlp/coffee_martini/
├── fourier_approx/coffee_martini/
├── polynomial_approx/coffee_martini/
└── neural_ode/coffee_martini/
```

Summary on `coffee_martini` (test set):

| Method | PSNR ↑ | SSIM ↑ | LPIPS-vgg ↓ | Flow EPE ↓ |
|---|---|---|---|---|
| **ODE-GS (ours)** | **28.73** | **0.921** | **0.105** | **1.42** |
| HexPlane + MLP | 28.21 | 0.923 | 0.116 | 1.61 |
| Neural ODE | 28.04 | 0.908 | 0.121 | 1.68 |
| Displacement MLP | 27.84 | 0.903 | 0.128 | 1.74 |
| Fourier Approx | 27.36 | 0.895 | 0.139 | 1.89 |
| Polynomial Approx | 26.92 | 0.887 | 0.151 | 2.04 |

---

## Installation

```bash
git clone <this-repo> && cd 4DGaussians_ODE3
git submodule update --init --recursive

conda create -n Gaussians4D python=3.8 && conda activate Gaussians4D

export CUDA_HOME=/usr/local/cuda-11.7
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

> Requires a CUDA-capable GPU with ≥ 8 GB VRAM.

---

## Running the Pipeline

### 1. Preprocess data

```bash
python preprocess_data.py                       # default: coffee_martini
python preprocess_data.py --dataset_name cook_spinach
python preprocess_data.py --source_dir /path/to/scene
```

Available scenes: `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`, `sear_steak`

### 2. Build point cloud (COLMAP)

```bash
bash multipleviewprogress.sh coffee_martini
```

### 3. Train

```bash
python train.py \
  -s data/multipleview/coffee_martini \
  -m output/multipleview/coffee_martini \
  --expname multipleview/coffee_martini \
  --model_class ode --iterations 30000 --dataloader
```

Replace `--model_class ode` with any method key from the table above to train a baseline.

### 4. Render

```bash
python render.py -m output/multipleview/coffee_martini \
  --model_class ode --skip_train --skip_video
```

### 5. Evaluate

```bash
python metrics.py -m output/multipleview/coffee_martini
python metrics.py -m output/multipleview/coffee_martini --eval_flow   # includes optical flow EPE
```

### 6. Run all methods at once

```bash
python full_eval.py                                          # all methods, coffee_martini
python full_eval.py --method ode                            # single method
python full_eval.py --skip_training --skip_rendering        # metrics only (results already exist)
```

---

## References

- Wu et al., *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*, CVPR 2024
- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG 2023

