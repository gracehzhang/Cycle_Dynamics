'''
Taken and  modified from the original robosuite repository (version 0.1.0)
Our fork with version 0.1.0 : https://github.com/eugval/robosuite
Official Robosuite Repository : https://github.com/ARISE-Initiative/robosuite

'''

import gym
import numpy as np
from collections import OrderedDict
import mujoco_py
from mujoco_py import MjSim, MjRenderContextOffscreen, MjViewer
from mujoco_py import load_model_from_xml

from gym.utils import seeding
from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer

REGISTERED_ENVS = {}

import time

def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def make(env_name, *args, **kwargs):
    """Try to get the equivalent functionality of gym.make in a sloppy way."""
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["MujocoEnv", "SawyerEnv", "BaxterEnv"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls


class MujocoEnv(gym.Env):
    """Initializes a Mujoco Environment."""

    def __init__(
        self,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        use_camera_obs=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        **kwargs
    ):
        """
        Args:

            has_renderer (bool): If true, render the simulation state in 
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes 
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes 
                in camera. False otherwise.

            control_freq (float): how many control signals to receive 
                in every simulated second. This sets the amount of simulation time 
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            use_camera_obs (bool): if True, every observation includes a 
                rendered image.

            camera_name (str): name of camera to be rendered. Must be 
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        self.seed()
        self._kwargs = kwargs
        self._screen_width = kwargs["screen_width"]
        self._screen_height = kwargs["screen_height"]

        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.control_freq = control_freq
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.viewer = None
        self.model = None

        # settings for camera observations
        self.use_camera_obs = use_camera_obs
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Camera observations require an offscreen renderer.")
        self.camera_name = camera_name
        if self.use_camera_obs and self.camera_name is None:
            raise ValueError("Must specify camera name when using camera obs")
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depth = camera_depth

        self.pid = None
        self._reset_internal()

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.
        """
        self.cur_time = 0
        self.model_timestep = self.sim.model.opt.timestep
        if self.model_timestep <= 0:
            raise XMLError("xml model defined non-positive time step")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError(
                "control frequency {} is invalid".format(control_freq)
            )
        self.control_timestep = 1. / control_freq

    def _setup_pid(self):
        "Function to setup the pid controller. Should set the self.pid attribute."
        raise NotImplementedError

    def _load_model(self):
        """Loads an xml model, puts it in self.model"""
        pass

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        pass

    def reset(self):
        """Resets simulation."""
        # if there is an active viewer window, destroy it
        self._destroy_viewer()
        self._reset_internal()
        self.sim.forward()
        return self._get_observation()

    def _reset_internal(self):
        """Resets simulation internal configurations."""
        # instantiate simulation from MJCF model
        self._load_model()
        self.mjpy_model = self.model.get_model(mode="mujoco_py")
        self.sim = MjSim(self.mjpy_model)
        self.initialize_time(self.control_freq)

        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = MjViewer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

            self.viewer.viewer._render_every_frame = True

        elif self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim)
                self.sim.add_render_context(render_context)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.sim._render_context_offscreen.vopt.geomgroup[1] = (
                1 if self.render_visual_mesh else 0
            )

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

    def _get_observation(self):
        """Returns an OrderedDict containing observations [(name_string, np.array), ...]."""
        return OrderedDict()

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        self._pre_action(action)
        end_time = self.cur_time + self.control_timestep
        while self.cur_time < end_time:
            if(self.pid is not None):
                self._set_pid_control()
            self.sim.step()
            self.cur_time += self.model_timestep
        reward, done, info = self._post_action(action)
        return self._get_observation(), reward, done, info

    def _set_pid_control(self):
        "Do any processing required with the pid"
        current_qvel = self.sim.data.qvel
        self.sim.data.ctrl[:] = self.pid(current_qvel, self.model_timestep)

    def _pre_action(self, action):
        """Do any preprocessing before taking an action."""
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        """Do any housekeeping after taking an action."""
        reward = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return reward, self.done, {}

    def reward(self, action):
        """Reward should be a function of state and action."""
        return 0

    def render(self, mode="rgb_array", height=100, width=100):
        """
        Renders to an off-screen window.
        """
        if height == None:
            height = self._screen_height
        if width == None:
            width = self._screen_width
        
        if mode == "rgb_array":
            camera_obs = self.sim.render(
                camera_name="birdview",
                width=width,
                height=height,
                depth=False,
            )
            #camera_obs = camera_obs[::-1, :, :] / 255.0
            assert np.sum(camera_obs) > 0, "rendering image is blank"
            return camera_obs
        elif mode == "human":
            self.viewer().render()
            return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def observation_spec(self):
        """
        Returns an observation as observation specification.

        An alternative design is to return an OrderedDict where the keys
        are the observation names and the values are the shapes of observations.
        We leave this alternative implementation commented out, as we find the
        current design is easier to use in practice.
        """
        observation = self._get_observation()
        # return observation

        observation_spec = OrderedDict()
        for k, v in observation.items():
            observation_spec[k] = v.shape
        return observation_spec

    def action_spec(self):
        """
        Action specification should be implemented in subclasses.

        Action space is represented by a tuple of (low, high), which are two numpy
        vectors that specify the min/max action limits per dimension.
        """
        raise NotImplementedError

    def reset_from_xml_string(self, xml_string):
        """Reloads the environment from an XML description of the environment."""

        # if there is an active viewer window, destroy it
        self.close()

        # load model from xml
        self.mjpy_model = load_model_from_xml(xml_string)

        self.sim = MjSim(self.mjpy_model)
        self.initialize_time(self.control_freq)
        if self.has_renderer and self.viewer is None:
            self.viewer = MjViewer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

        elif self.has_offscreen_renderer:
            render_context = MjRenderContextOffscreen(self.sim)
            render_context.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            render_context.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
            self.sim.add_render_context(render_context)

        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

        # necessary to refresh MjData
        self.sim.forward()

    def find_contacts(self, geoms_1, geoms_2):
        """
        Finds contact between two geom groups.

        Args:
            geoms_1: a list of geom names (string)
            geoms_2: another list of geom names (string)

        Returns:
            iterator of all contacts between @geoms_1 and @geoms_2
        """
        for contact in self.sim.data.contact[0 : self.sim.data.ncon]:
            # check contact geom in geoms
            c1_in_g1 = self.sim.model.geom_id2name(contact.geom1) in geoms_1
            c2_in_g2 = self.sim.model.geom_id2name(contact.geom2) in geoms_2
            # check contact geom in geoms (flipped)
            c2_in_g1 = self.sim.model.geom_id2name(contact.geom2) in geoms_1
            c1_in_g2 = self.sim.model.geom_id2name(contact.geom1) in geoms_2
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                yield contact

    def _check_contact(self):
        """Returns True if gripper is in contact with an object."""
        return False

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        return False

    def _destroy_viewer(self):
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close()  # change this to viewer.finish()?
            self.viewer = None

    def close(self):
        """Do any cleanup necessary here."""
        self._destroy_viewer()


    def get_state(self):
        return self.sim.get_state()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    def renderer_on(self):
        self.has_renderer = True

    def renderer_off(self):
        self.has_renderer = False
