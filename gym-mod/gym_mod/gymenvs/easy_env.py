import os

from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np

from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

from gym_mod.gymenvs.gym_walker_s2r_color_change import GymWalkerColor

class GymInvertedPendulumEasy(InvertedPendulumEnv):
    '''Color change, weight'''
    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum_easy.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

class GymWalkerEasy(GymWalkerColor):
    def __init__(self):
        self.step_count = 0
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2deasy.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)

class GymHalfCheetahEasy(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/half_cheetah_easy.xml")
        super().__init__(xml_file=model_path)
