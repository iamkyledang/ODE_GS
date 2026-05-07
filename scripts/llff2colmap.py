import os
import sys
import glob
import shutil
import numpy as np


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0.0, 0.0, 0.0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0.0, 0.0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0.0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0

    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_reference_image(cam_dir: str) -> str:
    """
    Prefer 0000.png, then 0000.jpg, otherwise first image found.
    """
    preferred = [
        os.path.join(cam_dir, "images", "0000.png"),
        os.path.join(cam_dir, "images", "0000.jpg"),
    ]
    for p in preferred:
        if os.path.isfile(p):
            return p

    candidates = sorted(
        glob.glob(os.path.join(cam_dir, "images", "*.png")) +
        glob.glob(os.path.join(cam_dir, "images", "*.jpg")) +
        glob.glob(os.path.join(cam_dir, "images", "*.jpeg"))
    )
    if not candidates:
        raise FileNotFoundError(f"No image found in {os.path.join(cam_dir, 'images')}")
    return candidates[0]


def llff_pose_to_colmap_extrinsic(pose_3x4: np.ndarray):
    """
    Convert LLFF-style pose to COLMAP world->camera extrinsic.

    Input pose_3x4 is assumed to come from:
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    followed by:
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

    Returns:
        qvec, tvec
    """
    R_c2w = pose_3x4[:, :3].copy()
    t_c2w = pose_3x4[:, 3].copy()

    # Keep the same convention adjustment as your original script.
    R_c2w = -R_c2w
    R_c2w[:, 0] = -R_c2w[:, 0]

    # COLMAP images.txt expects world-to-camera:
    # x_cam = R * x_world + t
    R_w2c = np.linalg.inv(R_c2w)
    t_w2c = -R_w2c @ t_c2w

    qvec = rotmat2qvec(R_w2c)
    return qvec, t_w2c


def main():
    if len(sys.argv) != 2:
        print("Usage: python llff2colmap.py <root_dir>")
        sys.exit(1)

    root_dir = sys.argv[1]
    poses_path = os.path.join(root_dir, "poses_bounds.npy")
    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"Missing poses file: {poses_path}")

    sparse_dir = os.path.join(root_dir, "sparse_")
    image_colmap_dir = os.path.join(root_dir, "image_colmap")

    # Clean old outputs to avoid stale files.
    if os.path.isdir(sparse_dir):
        shutil.rmtree(sparse_dir)
    if os.path.isdir(image_colmap_dir):
        shutil.rmtree(image_colmap_dir)

    ensure_dir(sparse_dir)
    ensure_dir(image_colmap_dir)

    poses_arr = np.load(poses_path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])   # (N, 3, 5)
    intrinsics = poses[:, :, -1]                    # (N, 3) = [H, W, focal]
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], axis=-1)

    cam_dirs = sorted(glob.glob(os.path.join(root_dir, "cam[0-9][0-9]")))
    if len(cam_dirs) != poses.shape[0]:
        raise ValueError(
            f"Number of camXX folders ({len(cam_dirs)}) does not match "
            f"number of poses ({poses.shape[0]})."
        )

    image_names = []
    image_paths = []

    for idx, cam_dir in enumerate(cam_dirs):
        src_img = find_reference_image(cam_dir)
        ext = os.path.splitext(src_img)[1].lower()
        dst_name = f"r_{idx:03d}{ext}"
        dst_path = os.path.join(image_colmap_dir, dst_name)

        shutil.copy2(src_img, dst_path)
        image_paths.append(src_img)
        image_names.append(dst_name)

    print("Copied reference images:")
    for p in image_paths:
        print(p)

    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    images_txt = os.path.join(sparse_dir, "images.txt")
    points3D_txt = os.path.join(sparse_dir, "points3D.txt")

    with open(cameras_txt, "w") as f_cam:
        f_cam.write("# Camera list with one line of data per camera:\n")
        f_cam.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f_cam.write(f"# Number of cameras: {len(poses)}\n")

        for idx in range(len(poses)):
            H, W, focal = intrinsics[idx]

            H, W, focal = intrinsics[idx]
            width = int(W)
            height = int(H)
            focal = float(focal)
            cx = width / 2.0
            cy = height / 2.0
            k = 0.0

            f_cam.write(
                f"{idx + 1} SIMPLE_RADIAL {width} {height} {focal} {cx} {cy} {k}\n"
            )

    with open(images_txt, "w") as f_img:
        f_img.write("# Image list with two lines of data per image:\n")
        f_img.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_img.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f_img.write(f"# Number of images: {len(poses)}\n")

        for idx, pose in enumerate(poses):
            qvec, tvec = llff_pose_to_colmap_extrinsic(pose)

            image_id = idx + 1
            camera_id = idx + 1
            name = image_names[idx]

            f_img.write(
                f"{image_id} "
                f"{qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{tvec[0]} {tvec[1]} {tvec[2]} "
                f"{camera_id} {name}\n"
            )
            f_img.write("\n")

    with open(points3D_txt, "w") as f_pts:
        f_pts.write("# 3D point list with one line of data per point:\n")
        f_pts.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f_pts.write("# Number of points: 0\n")

    print(f"\nWrote COLMAP text model to: {sparse_dir}")
    print(f"Wrote copied images to: {image_colmap_dir}")


if __name__ == "__main__":
    main()