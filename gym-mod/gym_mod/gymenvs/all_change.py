import os

from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np

from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv

from gym_mod.gymenvs.gym_walker_s2r_color_change import GymWalkerColor

class GymInvertedPendulumWeightColor(InvertedPendulumEnv):
    '''Color change, weight'''
    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum_weight_color.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)


class GymInvertedPendulumWeightColor2(InvertedPendulumEnv):
    '''Color change, viewpoint, weight'''
    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum_weight_color2.xml")
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

class GymInvertedPendulumWeightColor3(InvertedPendulumEnv):
    '''Color change, viewpoint, background, weight'''
    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum_weight_color3.xml")
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

class GymInvertedPendulumDM(InvertedPendulumEnv):
    '''Color change, viewpoint, background, weight'''
    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum_dm.xml")
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

DEFAULT_SIZE = 500

class GymHopperWeightColor(HopperEnv):
    '''Purple Hopper with White Floor, shiftview weight'''
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/hopper_weight_color.xml")
        super().__init__(xml_file=model_path)

    # def render(self,
    #            mode='human',
    #            width=DEFAULT_SIZE,
    #            height=DEFAULT_SIZE,
    #            camera_id=None,
    #            camera_name=None):
    #     import ipdb; ipdb.set_trace()
    #     return super().render(mode=mode, width=width, height=height, camera_id=camera_id, camera_name="asdf")


class GymWalkerWeightColor(GymWalkerColor):
    def __init__(self):
        self.step_count = 0
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2dweightcolor.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)
