#!/usr/bin/env bash
set -euo pipefail

workdir=data/multipleview/$1

bash colmap.sh "$workdir" llff

python scripts/downsample_point.py \
    "$workdir/colmap/dense/workspace/fused.ply" \
    "$workdir/points3D_multipleview.ply"



