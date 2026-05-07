"""
Preprocess Neural_3D_Video datasets for multipleviews scene format.

Source:  https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
Datasets available: coffee_martini, cook_spinach, cut_roasted_beef,
                    flame_salmon_1, flame_steak, sear_steak

Steps performed automatically:
  1. Download  <dataset_name>.zip from the GitHub release (skip if already present).
     flame_salmon_1 is split into 4 parts — all are downloaded and recombined.
  2. Unzip into --download_dir (default: current directory).
  3. Delete the zip file(s) after extraction.
  4. Renumber camera mp4 files to fill any numbering gaps
     (e.g. camera00, camera01, camera03 → camera00, camera01, camera02).
  5. Extract frames (every --stride frames) into data/multipleview/<dataset_name>/.
  6. Copy poses_bounds.npy into the output folder.

Output:  data/multipleview/<dataset_name>/
           cam00/images/frame_00001.jpg, frame_00002.jpg, ...
           cam01/images/frame_00001.jpg, ...
           ...
           poses_bounds.npy

Usage:
    python preprocess_data.py                                        # coffee_martini (default)
    python preprocess_data.py --dataset_name cook_spinach            # different dataset
    python preprocess_data.py --source_dir /path/to/existing/folder  # skip download
    python preprocess_data.py --stride 1                             # keep every frame

After running this script, execute:
    bash colmap.sh data/multipleview/<dataset_name> llff
"""

import os
import re
import shutil
import glob
import argparse
import subprocess
import urllib.request
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RELEASE_BASE = "https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0"
FLAME_SALMON  = "flame_salmon_1"

# flame_salmon_1 is a multi-part zip (z01 + z02 + z03 + zip = ~4.7 GB total)
FLAME_SALMON_PARTS = [
    "flame_salmon_1_split.z01",
    "flame_salmon_1_split.z02",
    "flame_salmon_1_split.z03",
    "flame_salmon_1_split.zip",
]

ALL_DATASETS = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: str) -> None:
    """Download url → dest with a simple MB progress indicator."""
    print(f"  Downloading {os.path.basename(dest)} ...")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct     = min(100.0, block_num * block_size * 100.0 / total_size)
            mb_done = block_num * block_size / 1_000_000
            mb_tot  = total_size / 1_000_000
            print(f"\r    {pct:5.1f}%  {mb_done:.0f} / {mb_tot:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


def download_and_extract(dataset_name: str, download_dir: str) -> str:
    """
    Download, extract, and clean up the zip for dataset_name.
    Returns the path to the extracted source folder containing the mp4 files.
    """
    os.makedirs(download_dir, exist_ok=True)

    if dataset_name == FLAME_SALMON:
        # ── Multi-part zip ────────────────────────────────────────────────
        for fname in FLAME_SALMON_PARTS:
            dest = os.path.join(download_dir, fname)
            if not os.path.exists(dest):
                _download_file(f"{RELEASE_BASE}/{fname}", dest)
            else:
                print(f"  Already downloaded, skipping: {fname}")

        combined_zip = os.path.join(download_dir, "flame_salmon_1.zip")
        if not os.path.exists(combined_zip):
            print("  Combining split parts with `zip -F` ...")
            split_zip = os.path.join(download_dir, "flame_salmon_1_split.zip")
            subprocess.run(
                ["zip", "-F", split_zip, "--out", combined_zip],
                check=True,
            )

        _unzip(combined_zip, download_dir)

        # Clean up all split parts and the combined zip
        for fname in FLAME_SALMON_PARTS:
            _rm(os.path.join(download_dir, fname))
        _rm(combined_zip)

    else:
        # ── Single zip ────────────────────────────────────────────────────
        zip_name = f"{dataset_name}.zip"
        zip_path = os.path.join(download_dir, zip_name)
        if not os.path.exists(zip_path):
            _download_file(f"{RELEASE_BASE}/{zip_name}", zip_path)
        else:
            print(f"  Already downloaded, skipping: {zip_name}")

        _unzip(zip_path, download_dir)
        _rm(zip_path)

    # The zip should have extracted to a subfolder named after the dataset
    extracted = os.path.join(download_dir, dataset_name)
    if not os.path.isdir(extracted):
        # Fallback: maybe it extracted files directly into download_dir
        extracted = download_dir

    return extracted


def _unzip(zip_path: str, dest_dir: str) -> None:
    extracted_name = os.path.splitext(os.path.basename(zip_path))[0]
    out_folder = os.path.join(dest_dir, extracted_name.replace("_split", ""))
    if os.path.isdir(out_folder):
        print(f"  Already extracted, skipping unzip: {os.path.basename(zip_path)}")
        return
    print(f"  Unzipping {os.path.basename(zip_path)} ...")
    subprocess.run(["unzip", "-q", zip_path, "-d", dest_dir], check=True)
    print(f"  Extracted to {dest_dir}")


def _rm(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"  Deleted {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Camera renumbering
# ─────────────────────────────────────────────────────────────────────────────

def renumber_cameras(source_dir: str) -> None:
    """
    Rename camera mp4 files to fill any numbering gaps.

    Example: camera00.mp4, camera01.mp4, camera03.mp4 (missing 02)
             → camera00.mp4, camera01.mp4, camera02.mp4
    """
    # Collect all camera mp4 files (cam* or camera*)
    mp4_files = (
        glob.glob(os.path.join(source_dir, "camera*.mp4")) +
        glob.glob(os.path.join(source_dir, "cam*.mp4"))
    )
    if not mp4_files:
        return

    # Extract (number, filepath) pairs
    numbered = []
    for fp in mp4_files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        m = re.search(r"(\d+)$", stem)
        if m:
            numbered.append((int(m.group(1)), fp))

    if not numbered:
        return

    numbered.sort()
    numbers = [n for n, _ in numbered]
    expected = list(range(len(numbers)))

    if numbers == expected:
        return  # already sequential, nothing to do

    print(f"\n  Renumbering cameras: found indices {numbers}")

    # Detect shared prefix (e.g. "camera" or "cam")
    first_stem = os.path.splitext(os.path.basename(numbered[0][1]))[0]
    prefix = re.sub(r"\d+$", "", first_stem)

    # Step 1 – rename to temp names to avoid collisions
    temps = []
    for i, (_, fp) in enumerate(numbered):
        tmp = fp + ".renaming_tmp"
        os.rename(fp, tmp)
        temps.append((i, tmp))

    # Step 2 – rename to final sequential names
    for i, tmp in temps:
        new_path = os.path.join(source_dir, f"{prefix}{i:02d}.mp4")
        os.rename(tmp, new_path)
        old_stem = os.path.splitext(os.path.basename(tmp[: -len(".renaming_tmp")]))[0]
        print(f"    {old_stem}.mp4 → {prefix}{i:02d}.mp4")


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, out_dir: str, stride: int = 1, ext: str = "jpg") -> int:
    """
    Extract frames from a video into out_dir, saving every stride-th frame.
    Filenames are 1-indexed: frame_00001.jpg, frame_00002.jpg, ...
    Returns the number of frames saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            out_path = os.path.join(out_dir, f"frame_{saved_idx + 1:05d}.{ext}")
            if ext.lower() in ("jpg", "jpeg"):
                cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return saved_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(
    dataset_name: str,
    output_base: str,
    stride: int,
    ext: str,
    source_dir: str,
    download_dir: str,
) -> None:
    # ── Step 1: Download (if source_dir not already provided / doesn't exist) ──
    if source_dir is None or not os.path.isdir(source_dir):
        if source_dir is not None:
            print(f"  source_dir {source_dir!r} not found — downloading instead.")
        print(f"\n[download] {dataset_name}")
        source_dir = download_and_extract(dataset_name, download_dir)

    print(f"\n[source] {source_dir}")

    # ── Step 2: Renumber cameras if there are gaps ─────────────────────────
    print(f"\n[renumber] Checking camera numbering in {source_dir} ...")
    renumber_cameras(source_dir)

    # ── Step 3: Extract frames ─────────────────────────────────────────────
    output_dir = os.path.join(output_base, "multipleview", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Accept both cam*.mp4 and camera*.mp4 naming conventions
    video_files = sorted(
        glob.glob(os.path.join(source_dir, "camera*.mp4")) +
        glob.glob(os.path.join(source_dir, "cam*.mp4"))
    )
    if not video_files:
        raise FileNotFoundError(
            f"No camera mp4 files found in {source_dir}\n"
            f"Expected files matching cam*.mp4 or camera*.mp4"
        )

    print(f"\n[extract] {len(video_files)} camera(s) → {output_dir}")
    for i, video_path in enumerate(video_files):
        cam_name   = f"cam{i:02d}"
        images_out = os.path.join(output_dir, cam_name, "images")
        print(f"  {os.path.basename(video_path)} → {cam_name}/images/ ...", end=" ", flush=True)
        n = extract_frames(video_path, images_out, stride=stride, ext=ext)
        print(f"{n} frames saved.")

    # ── Step 4: Copy poses_bounds.npy ──────────────────────────────────────
    poses_src = os.path.join(source_dir, "poses_bounds.npy")
    poses_dst = os.path.join(output_dir, "poses_bounds.npy")
    if os.path.exists(poses_src):
        shutil.copy2(poses_src, poses_dst)
        print(f"\n[copy] poses_bounds.npy → {poses_dst}")
    else:
        print(f"\n[warning] poses_bounds.npy not found in {source_dir}, skipping.")

    print(f"\n[done] Dataset ready at: {output_dir}")
    print(f"Next step: bash colmap.sh {output_dir} llff")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preprocess Neural_3D_Video datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name", default="coffee_martini", choices=ALL_DATASETS,
        help="Which dataset to download and preprocess.",
    )
    parser.add_argument(
        "--source_dir", default=None,
        help="Path to an already-downloaded/extracted folder containing cam*.mp4 files. "
             "If omitted (or folder doesn't exist), the dataset is downloaded automatically.",
    )
    parser.add_argument(
        "--download_dir", default=".",
        help="Directory where the zip is downloaded and extracted before processing.",
    )
    parser.add_argument(
        "--output_base", default="data",
        help="Base output directory; frames go to <output_base>/multipleview/<dataset_name>/.",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Save every Nth frame (1 = keep all frames).",
    )
    parser.add_argument(
        "--ext", default="jpg", choices=["jpg", "png"],
        help="Output image format.",
    )
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        output_base=args.output_base,
        stride=args.stride,
        ext=args.ext,
        source_dir=args.source_dir,
        download_dir=args.download_dir,
    )
