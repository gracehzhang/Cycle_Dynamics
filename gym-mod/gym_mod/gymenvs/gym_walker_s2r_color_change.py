from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import os


class GymWalkerColor(Walker2dEnv):
    def __init__(self):
        self.step_count = 0
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2dcolor.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ob, reward, done, x = super(GymWalkerColor, self).step(a)
        self.step_count += 1
        return ob, reward, done, x

    def reset_model(self):
        self.step_count = 0
        obs = super(GymWalkerColor, self).reset_model()
        return obs

class GymWalkerColorNoBackground(GymWalkerColor):
    def __init__(self):
        self.step_count = 0
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2dcolornobackground.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)

class GymWalkerDM(GymWalkerColor):
    def __init__(self):
        self.step_count = 0
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_dm.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)
