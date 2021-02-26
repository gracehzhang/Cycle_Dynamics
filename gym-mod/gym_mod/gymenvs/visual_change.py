import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv


class GymHalfCheetahColor(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/half_cheetah_color.xml")
        super().__init__(xml_file=model_path)

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        # Backwards Reward
        reward = - forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

class GymHalfCheetahDM(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/half_cheetah_dm.xml")
        super().__init__(xml_file=model_path)

class GymHopperColor(HopperEnv):
    '''Purple Hopper'''
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/hopper_color.xml")
        super().__init__(xml_file=model_path)

class GymHopperColor2(HopperEnv):
    '''Purple Hopper with White Floor'''
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/hopper_color2.xml")
        super().__init__(xml_file=model_path)

    def viewer_setup(self):
        # DEFAULT_CAMERA_CONFIG = {
        #     'trackbodyid': 2,
        #     'distance': 3.0,
        #     'lookat': np.array((0.0, 0.0, 1.15)),
        #     'elevation': -20.0,
        # }
        # DEFAULT_CAMERA_CONFIG = {
        #     'elevation': -20.0,
        #     'azimuth': 90,
        # }
        super().viewer_setup()
        DEFAULT_CAMERA_CONFIG = {
            'elevation': -30.0,
            'azimuth': 100.0,
            'distance': 4.0,
            'trackbodyid': 1,
        }

        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class GymInvertedPendulumColor(InvertedPendulumEnv):
    '''Purple/Orange/Blue cart pole'''
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum_color.xml")
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

class GymInvertedPendulumBackground(InvertedPendulumEnv):
    '''Purple/Orange/Blue cart pole with different background'''
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum_background.xml")
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

class GymInvertedPendulumViewpoint(InvertedPendulumEnv):
    '''cart pole from a different viewpoint'''
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum_viewpoint.xml")
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

    def viewer_setup(self):
        # DEFAULT_CAMERA_CONFIG = {
        #     'elevation': -55.0,
        #     'lookat': np.array([0.05, 0.0, 0.0]),
        # }
        DEFAULT_CAMERA_CONFIG = {
            'elevation': -55.0,
            'azimuth': 100.0,
        }
        # import ipdb
        # ipdb.set_trace()
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class GymReacherColor(ReacherEnv):

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/reacher_color.xml")
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

class GymReacher(ReacherEnv):
    '''Default reacher with larger fingertip and target'''
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/reacher.xml")
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)
