import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo


# ---------------------------------------------------------------------------
# Spiral camera path generation (previously in neural_3D_dataset_NDC.py)
# ---------------------------------------------------------------------------

def _normalize(v):
    return v / np.linalg.norm(v)


def _average_poses(poses):
    center = poses[:, :3, 3].mean(0)
    z = _normalize(poses[:, :3, 2].mean(0))
    y_ = poses[:, :3, 1].mean(0)
    x = _normalize(np.cross(y_, z))
    y = np.cross(z, x)
    return np.stack([x, y, z, center], 1)  # (3, 4)


def _render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    for theta in np.linspace(0.0, 2.0 * np.pi * zrate, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads,
        )
        z = _normalize(c - np.dot(c2w[:3, :4], np.array([0.0, 0.0, -focal, 1.0])))
        render_poses.append(np.stack([_normalize(np.cross(up, z)), up, z, c]))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """Generate a spiral render path around the scene.

    Args:
        c2ws_all:   (N, 3, 4) or (N, 3, 5) camera-to-world matrices.
        near_fars:  (N, 2) per-camera near/far distances.
        rads_scale: scale factor applied to the spiral radii.
        N_views:    number of poses to generate.

    Returns:
        (N_views, 3, 4) render poses.
    """
    c2w = _average_poses(c2ws_all)
    up = _normalize(c2ws_all[:, :3, 1].sum(0))
    dt = 0.75
    close_depth = near_fars.min() * 0.9
    inf_depth   = near_fars.max() * 5.0
    focal  = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    zdelta = close_depth * 0.2
    rads   = np.percentile(np.abs(c2ws_all[:, :3, 3]), 90, axis=0) * rads_scale
    return np.array(_render_path_spiral(c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views))
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split
    ):
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)
        
    
    def load_images_path(self, cam_folder, cam_extrinsics,cam_intrinsics,split):
        image_length = len(os.listdir(os.path.join(cam_folder,"cam01","images")))
        #len_cam=len(cam_extrinsics)
        image_paths=[]
        image_poses=[]
        image_times=[]
        self.image_cam_names = []
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            number = os.path.basename(extr.name)[5:-4]
            cam_name = "cam" + number.zfill(2)
            images_folder=os.path.join(cam_folder,"cam"+number.zfill(2),"images")

            image_range=range(image_length)
            if split=="test":
                image_range = range(200, image_length)
            elif split=="train":
                image_range = range(min(200, image_length))

            for i in image_range:    
                num=i+1
                image_path=os.path.join(images_folder,"frame_"+str(num).zfill(5)+".jpg")
                image_paths.append(image_path)
                image_poses.append((R,T))
                image_times.append(float(i/image_length))
                self.image_cam_names.append(cam_name)

        return image_paths, image_poses,image_times
    
    def get_video_cam_infos(self,datadir):
        poses_path = os.path.join(datadir, "poses_bounds_multipleview.npy")
        if not os.path.exists(poses_path):
            poses_path = os.path.join(datadir, "poses_bounds.npy")
        poses_arr = np.load(poses_path)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            p = np.array(p)
            if p.shape == (4, 3):   # _render_path_spiral returns (4,3); transpose to (3,4)
                p = p.T
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]