# 4D Gaussian Splatting with ODE-Based Deformation

Each Gaussian primitive evolves through **closed-form ODEs** — no neural deformation field. The original 3DGS rasterizer is preserved; only the temporal parameterisation changes. Includes the proposed **ODE-GS** method and five baselines, all trained and evaluated under identical conditions on the [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) dataset.

---

## Method

Per-primitive ODE parameters (`A`, `b`, `x₀`, `κ`, `ω`) drive mean and covariance analytically:

```
μ_i(τ)  = μ_i⁰ + exp([A_i, b_i; 0, 0] τ) x₀_i    (4×4 matrix exp)
s_j(τ)  = s_j⁰ + κ_j τ                              (log-scale)
R(τ)    = exp(ω̂ τ) R⁰                               (SO(3))
```

Loss: `L = L_rec + λ_f L_flow + λ_r L_ode`

---

## Baselines

| Key | Method | Reference |
|---|---|---|
| `ode` | **ODE-GS** (this work) | — |
| `deformable_mlp` | Displacement MLP | Deformable-3D-Gaussians (CVPR 2024) |
| `deformable_hexplane_mlp` | HexPlane + MLP | 4DGaussians (CVPR 2024) |
| `fourier_approx` | Fourier series (K=2) | EfficientDynamic3DGaussian |
| `polynomial_approx` | Polynomial (degree 3) | Gaussian-Flow |
| `neural_ode` | Velocity MLP + Euler integration | evogs |

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

> GPU VRAM ≥ 8 GB required. Hardware (RTX 4090 / other) is detected automatically.

---

## Pipeline

### 1. Preprocess data

```bash
python preprocess_data.py                                   # coffee_martini
python preprocess_data.py --dataset_name cook_spinach
python preprocess_data.py --source_dir /path/to/scene      # pre-downloaded
python preprocess_data.py --stride 5                        # every 5th frame
```

Scenes: `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`, `sear_steak`.
Output: `data/multipleview/<scene>/cam*/images/frame_*.jpg` + `poses_bounds.npy`

### 2. COLMAP point cloud

```bash
bash multipleviewprogress.sh coffee_martini
# → data/multipleview/coffee_martini/points3D_multipleview.ply
```

### 3. Train

```bash
python train.py \
  -s data/multipleview/coffee_martini \
  -m output/multipleview/coffee_martini \
  --expname multipleview/coffee_martini \
  --model_class ode --iterations 30000 --dataloader
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--model_class` | `ode` | One of the six method keys above |
| `--iterations` | `30000` | Fine-stage iterations |
| `--coarse_iterations` | `3000` | Canonical-only warm-up |
| `--lambda_ode` | `0.001` | ODE regularisation weight |
| `--lambda_dssim` | `0.2` | SSIM loss weight |
| `--ode_lr_init` | `0.001` | ODE parameter learning rate |
| `--dataloader` | off | Enable multi-CPU DataLoader |

### 4. Render

```bash
python render.py -m output/multipleview/coffee_martini \
  --model_class ode --skip_train --skip_video
# → output/multipleview/coffee_martini/test/ours_<iter>/renders/ and gt/
```

### 5. Metrics

```bash
python metrics.py -m output/multipleview/coffee_martini
python metrics.py -m output/multipleview/coffee_martini --eval_flow   # + optical flow EPE
```

Outputs: `results.json`, `per_view.json`, `visualizations/test/<ours_N>/`
Metrics: PSNR, SSIM, LPIPS-vgg, LPIPS-alex, MS-SSIM, D-SSIM

---

## Full Baseline Comparison

`full_eval.py` runs train → render → metrics for all methods. Each phase is skipped if output already exists.

```bash
python full_eval.py                                          # all methods, coffee_martini
python full_eval.py --method ode                            # single method
python full_eval.py --methods ode fourier_approx neural_ode # subset
python full_eval.py --skip_training                         # render + metrics only
python full_eval.py --skip_training --skip_rendering        # metrics + table only
python full_eval.py --scenes cook_spinach
python full_eval.py --data_root /data/neural3d --output_root /results
```

Output: `output/output_baselines/result.json` + `comparison_table.png` + per-method subdirs.

---

## Citation

```bibtex
@InProceedings{Wu_2024_CVPR,
    author = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and others},
    title  = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
    booktitle = {CVPR}, year = {2024},
}
@article{kerbl3Dgaussians,
    author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal = {ACM Transactions on Graphics}, volume = {42}, number = {4}, year = {2023},
}
```

