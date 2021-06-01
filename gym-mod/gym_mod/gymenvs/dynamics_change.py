import os

from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym_mod.gymenvs.gym_walker_s2r_color_change import GymWalkerColor


class GymInvertedPendulumWeight(InvertedPendulumEnv):
    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum_weight.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)


class GymHopperWeight(HopperEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/hopper_weight.xml")
        super().__init__(xml_file=model_path)

class GymHalfCheetahArmature(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/half_cheetah_armature.xml")
        super().__init__(xml_file=model_path)

class GymWalkerWeight(GymWalkerColor):
    def __init__(self):
        self.step_count = 0
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2dweight.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)
