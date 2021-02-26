from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.mujoco.wrapper import mjbindings
from dm_control.suite import base
from dm_control.suite import walker
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards


_DEFAULT_TIME_LIMIT = 10
_CONTROL_TIMESTEP = 0.01

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8


SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("walker.xml"), common.ASSETS


@SUITE.add("benchmarking")
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = DMWalker(move_speed=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


@SUITE.add("benchmarking")
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = DMWalker(move_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


@SUITE.add("benchmarking")
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = DMWalker(move_speed=_RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


def small_randomize(physics, random):
    """custumized radomize function to initialize 
    joint angle from a small range"""
    random = random or np.random

    hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
    slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
    ball = mjbindings.enums.mjtJoint.mjJNT_BALL
    free = mjbindings.enums.mjtJoint.mjJNT_FREE

    qpos = physics.named.data.qpos

    for joint_id in range(physics.model.njnt):
        joint_name = physics.model.id2name(joint_id, "joint")
        joint_type = physics.model.jnt_type[joint_id]
        is_limited = physics.model.jnt_limited[joint_id]
        range_min, range_max = physics.model.jnt_range[joint_id]

        if is_limited:
            if joint_type == hinge or joint_type == slide:
                qpos[joint_name] += random.uniform(-0.005, 0.005)
        else:
            if joint_type == hinge:
                qpos[joint_name] += random.uniform(-0.005, 0.005)


class DMWalker(walker.PlanarWalker):
    """A planar walker task initilized differently from suit"""

    def initialize_episode(self, physics):
        small_randomize(physics, self.random)
        # super(DMWalker, self).initialize_episode(physics)
