"""
Preprocess coffee_martini for multipleviews scene format.

Source:  coffee_martini/cam*.mp4  (+ poses_bounds.npy)
Output:  data/multipleview/coffee_martini/
           cam00/images/frame_00001.jpg, frame_00002.jpg, ...
           cam01/images/frame_00001.jpg, ...
           ...
           poses_bounds.npy  (copied from source)

This script supports temporal subsampling via --stride (default 5) to
reduce the number of saved frames and the storage/memory footprint.

After running this script, execute:
    bash colmap.sh data/multipleview/coffee_martini llff
"""

import os
import shutil
import glob
import argparse
import cv2


def extract_frames(video_path: str, out_dir: str, stride: int = 5, ext: str = "jpg") -> int:
    """Extract frames from a video into out_dir, saving every `stride`-th frame.

    Filenames are 1-indexed and follow frame_{:05d}.{ext} to match the
    project's expected naming (e.g. frame_00001.jpg).
    Returns the number of frames saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0   # index of frames read from the video (0-based)
    saved_idx = 0   # number of frames saved so far
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            out_path = os.path.join(out_dir, f"frame_{saved_idx+1:05d}.{ext}")
            # write as JPEG by default for smaller size
            if ext.lower() in ("jpg", "jpeg"):
                cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return saved_idx


def main(source_dir: str, output_base: str, dataset_name: str, stride: int = 5, ext: str = "jpg") -> None:
    output_dir = os.path.join(output_base, "multipleview", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted(glob.glob(os.path.join(source_dir, "cam*.mp4")))
    if not video_files:
        raise FileNotFoundError(f"No cam*.mp4 files found in {source_dir}")

    print(f"Found {len(video_files)} camera video(s). Extracting frames to: {output_dir}\n")

    for video_path in video_files:
        cam_name = os.path.splitext(os.path.basename(video_path))[0]  # e.g. "cam00"
        images_out = os.path.join(output_dir, cam_name, "images")
        print(f"  {cam_name}: extracting frames (every {stride} frames) ...", end=" ", flush=True)
        n = extract_frames(video_path, images_out, stride=stride, ext=ext)
        print(f"{n} frames saved.")

    # Copy poses_bounds.npy to output dir (llff2colmap.py reads it from workdir root)
    poses_src = os.path.join(source_dir, "poses_bounds.npy")
    poses_dst = os.path.join(output_dir, "poses_bounds.npy")
    if os.path.exists(poses_src):
        shutil.copy2(poses_src, poses_dst)
        print(f"\nCopied poses_bounds.npy -> {poses_dst}")
    else:
        print(f"\nWarning: poses_bounds.npy not found in {source_dir}, skipping.")

    print(f"\nDone. Dataset ready at: {output_dir}")
    print("Next step: bash colmap.sh data/multipleview/coffee_martini llff")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess coffee_martini for multipleviews format.")
    parser.add_argument("--source_dir", default="coffee_martini",
                        help="Path to the source folder containing cam*.mp4 files.")
    parser.add_argument("--output_base", default="data",
                        help="Base output directory (default: data/).")
    parser.add_argument("--dataset_name", default="coffee_martini",
                        help="Name of the dataset subfolder under multipleview/.")
    parser.add_argument("--stride", type=int, default=5,
                        help="Save every Nth frame (default: 5).")
    parser.add_argument("--ext", default="jpg",
                        help="Output image extension/format (jpg or png).")
    args = parser.parse_args()

    main(
        source_dir=args.source_dir,
        output_base=args.output_base,
        dataset_name=args.dataset_name,
        stride=args.stride,
        ext=args.ext,
    )
