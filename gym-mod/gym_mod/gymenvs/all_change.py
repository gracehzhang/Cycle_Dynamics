import os

from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import mujoco_py

from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv

from gym_mod.gymenvs.gym_walker_s2r_color_change import GymWalkerColor
from gym.envs.robotics.fetch.reach import FetchReachEnv
from gym.envs.robotics import fetch_env
from gym.envs.robotics import utils as fetch_utils
from gym import error, spaces

class GymFetchReachEnv(FetchReachEnv):
    '''Unity'''
    def __init__(self, xml_path="reach_target.xml", **kwargs):
        self._screen_width = 100
        self._screen_height = 100

        theta = 45 * np.pi / 180 # pi/2 = 90 deg
        c, s = np.cos(theta), np.sin(theta)
        self.rot = np.array(((c, -s), (s, c)))
        self.bias = np.array([0,0,-0.5,0]) #np.array([0,0,-0.5,0])

        super().__init__(xml_path=os.path.join("fetch", xml_path))


    def step(self, action):
        obs, rew, done, info  = super().step(action)

        # obs = np.concatenate([v for (k,v) in obs.items()])
        return obs, rew, done, info

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.0
        self.viewer.cam.azimuth = 177 #132.
        self.viewer.cam.elevation = -32 #14.

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        ### action bias

        action[:2] = self.rot.dot(action[:2])
        action = action + self.bias

        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        fetch_utils.ctrl_set_action(self.sim, action)
        fetch_utils.mocap_set_action(self.sim, action)

class GymFetchReach2Env(GymFetchReachEnv):
    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.2
        self.viewer.cam.azimuth = 165 #132.
        self.viewer.cam.elevation = -35 #14.


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
