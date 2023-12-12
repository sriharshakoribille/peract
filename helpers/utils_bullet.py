import numpy as np
import pickle
from helpers.observation_bullet import Observation_bullet

remove_keys = ['intrinsics','extrinsics']

def extract_obs(obs: Observation_bullet,
				cameras,
                t: int = 0,
                prev_action = None,
                channels_last: bool = False,
                episode_length: int = 10):

    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)

    obs_dict = {}
    robot_state = np.array([
                  obs.gripper_open,
                  *obs.gripper_joint_positions])
    

    for camera_name in cameras:
        cam = obs.cams[camera_name]
        cam_dict = {k: v for k, v in vars(cam).items() if k not in remove_keys}
        if not channels_last:
            # swap channels from last dim to 1st dim
            cam_dict = {camera_name +'_'+ k: np.transpose(
                v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                        for k, v in cam_dict.items()}
        else:
            # add extra dim to depth data
            cam_dict = {camera_name +'_'+ k: v if v.ndim == 3 else np.expand_dims(v, -1)
                        for k, v in cam_dict.items()}
        obs_dict.update(cam_dict)
        # obs_dict['%s_camera_extrinsics' % camera_name] = cam.extrinsics
        # obs_dict['%s_camera_intrinsics' % camera_name] = cam.intrinsics

    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    # add timestep to low_dim_state
    # episode_length = 10 
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    obs_dict['low_dim_state'] = np.concatenate(
        [obs_dict['low_dim_state'], [time]]).astype(np.float32)

    return obs_dict

def extract_demo(episode, save_dir, desc=False, kps=False):
    if desc:
        demo_path = '/episode_%d/desc.pkl'%episode
    elif kps:
        demo_path = '/episode_%d/kps.pkl'%episode
    else:
        demo_path = '/episode_%d/obs.pkl'%episode
    path = save_dir + demo_path
    # print(path)
    with open(path, 'rb') as f:
        demo = pickle.load(f)
    return demo