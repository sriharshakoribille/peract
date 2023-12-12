import numpy as np
from typing import Type

def construct_pointcloud(
        depth: np.ndarray, extrinsics: np.ndarray,
        intrinsics: np.ndarray) -> np.ndarray:
    """Converts depth (in meters) to point cloud in word frame.
    :return: A numpy array of size (width, height, 3)
    """
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(
        pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]
    return world_coords

def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                      (h, w, -1))

def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo

class Camera_Observation(object):
    """Storage for camera observations."""
    def __init__(self, rgb, depth, intrinsics, extrinsics) -> None:
        self.rgb = rgb
        self.depth = depth
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.point_cloud = None
    
    def add_point_cloud(self):
        cam_corrective_T = np.array([[ 0, 0, 1, 0], 
                  [-1, 0, 0, 0], 
                  [ 0,-1, 0, 0],
                  [ 0, 0, 0, 1]])
        res_frames = np.min(self.depth.shape)
        depth_cam = self.depth[0:res_frames,0:res_frames]
        extr = self.extrinsics @ cam_corrective_T
        self.point_cloud = construct_pointcloud(depth_cam,extr,self.intrinsics)

class Observation_bullet(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 joint_velocities: np.ndarray,
                 joint_positions: np.ndarray,
                 joint_torques: np.ndarray,
                 gripper_open: float,
                 gripper_pose: dict,
                 cams: dict,
                 ignore_collisions=0,
                 misc=None):
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_torques = joint_torques
        self.gripper_open = gripper_open
        self.gripper_pose = [*gripper_pose['position'], *gripper_pose['quaternion']]
        self.gripper_joint_positions = joint_positions[-2:]
        # self.ignore_collisions = ignore_collisions
        self.cams = {cam: Camera_Observation(*params) for cam,params in cams.items()}
        self.misc = misc
        self.fill_point_clouds()
    
    def get_frames(self, cam):
        return self.cams[cam].rgb, self.cams[cam].depth
    
    def fill_point_clouds(self):
        for cam in self.cams.values():
            cam.add_point_cloud()

class Demo_bullet(object):

    def __init__(self, observations, random_seed=None):
        self._observations = [Observation_bullet(**obs) for obs in observations]
        self.random_seed = random_seed
        self.variation_number = 0

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)