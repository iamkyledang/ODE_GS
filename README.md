# Dynamic 4D Gaussian Splatting with ODE-Based Deformation

Extends [4D Gaussian Splatting (Wu et al., CVPR 2024)](https://guanjunwu.github.io/4dgs/index.html) by replacing the HexPlane + MLP deformation with explicit per-Gaussian ODE parameters for mean and covariance evolution.

---

## Installation

Please follow the instruction bellow for installation

```bash
git clone https://github.com/iamkyledang/ODE_GS
cd ODE_GS
git submodule update --init --recursive
conda create -n Gaussians4D python=3.7
conda activate Gaussians4D
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

Then clone this repo and install:

```bash
git clone <this-repo> && cd 4DGaussians_ODE3
git submodule update --init --recursive

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

Downloads the dataset from the [Neural_3D_Video release](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0), extracts frames, and places them in `data/multipleview/<dataset_name>/`.

Available datasets: `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`, `sear_steak`.

```bash
# Default: coffee_martini, all frames (stride=1)
python preprocess_data.py

# Different dataset
python preprocess_data.py --dataset_name cook_spinach

# Already downloaded — point to existing folder
python preprocess_data.py --source_dir /path/to/coffee_martini

# Subsample frames (e.g. every 5th frame)
python preprocess_data.py --stride 5
```

Output layout:
```
data/multipleview/coffee_martini/
  cam00/images/frame_00001.jpg ...
  cam01/images/frame_00001.jpg ...
  poses_bounds.npy
```

> Camera files are renumbered automatically if there are any gaps (e.g. camera00, camera01, camera03 → camera00, camera01, camera02).

---

### Step 2 — COLMAP & Point Cloud

Runs COLMAP and downsamples the point cloud for training.

```bash
bash multipleviewprogress.sh coffee_martini
```

This produces `data/multipleview/coffee_martini/points3D_multipleview.ply`.

---

### Step 3 — Train

```bash
python train.py \
  -s data/multipleview/coffee_martini \
  -m output/multipleview/coffee_martini \
  --expname multipleview/coffee_martini \
  --dataloader
```

Add `--iterations 100000` for a full run (default is 30,000). The train split is the first 200 frames per camera; the test split is the remaining frames.

Key flags:

| Flag | Description | Default |
|---|---|---|
| `--model_class` | `ode`, `deformable_mlp`, `deformable_hexplane_mlp`, `fourier_approx`, `polynomial_approx` | `ode` |
| `--iterations` | Total training iterations | `30000` |
| `--dataloader` | Use DataLoader (recommended for multi-view) | off |
| `--lambda_dssim` | SSIM loss weight | `0.2` |

---

### Step 4 — Render

Renders the **test split** only.

```bash
python render.py \
  -m output/multipleview/coffee_martini \
  --model_class ode \
  --skip_train \
  --skip_video
```

Rendered frames and GT images are saved to `output/multipleview/coffee_martini/test/ours_<iteration>/`.

---

### Step 5 — Metrics

Evaluates PSNR, SSIM, LPIPS-vgg, LPIPS-alex, MS-SSIM, D-SSIM on the test split.

```bash
python metrics.py -m output/multipleview/coffee_martini
```

Results are written to `output/multipleview/coffee_martini/results.json`. Works for any model class — just point it at the output directory.

---

## Baseline Comparison (all methods at once)

Trains, renders, and evaluates all five methods on coffee_martini, then produces a comparison table and image.

```bash
python full_eval.py
```

Common options:

```bash
# Run only a subset of methods
python full_eval.py --methods ode fourier_approx

# Skip training (use existing checkpoints)
python full_eval.py --skip_training

# Skip training and rendering, recompute metrics + table only
python full_eval.py --skip_training --skip_rendering

# Different scene
python full_eval.py --scenes cook_spinach
```

Methods compared: `ode`, `deformable_mlp`, `deformable_hexplane_mlp`, `fourier_approx`, `polynomial_approx`.

Outputs saved to `output/output_baselines/`:
- `result.json` — merged metrics for all methods
- `comparison_table.png` — visual comparison table

---

## Citation

```bibtex
@InProceedings{Wu_2024_CVPR,
    author    = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and
                 Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
    title     = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
    booktitle = {CVPR},
    year      = {2024},
}
```

