from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import os


class GymWalker(Walker2dEnv):
    def __init__(self):
        self.step_count = 0
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ob, reward, done, x = super(GymWalker, self).step(a)
        self.step_count += 1
        return ob, reward, done, x

    def reset_model(self):
        self.step_count = 0
        obs = super(GymWalker, self).reset_model()
        return obs

class GymWalkerBackwards(GymWalker):

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = -1.0 * ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        self.step_count += 1
        return ob, reward, done, {}

