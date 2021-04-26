""" Define all custom gym_mod. """

from gym.envs.registration import register
#from dm_control import suite
#from gym_mod.dmenvs import dm_walker_s2r

register(
    id="GymFetchReach-v0",
    entry_point="gym_mod.gymenvs:GymFetchReachEnv",
    kwargs={},
    max_episode_steps=50
)

register(
    id="GymFetchReach2-v0",
    entry_point="gym_mod.gymenvs:GymFetchReachEnv",
    kwargs={},
    max_episode_steps=50
)

register(
    id="SawyerPushShiftViewZoomBackground-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushShiftViewZoomBackground",
    kwargs={},
    max_episode_steps=70
)

register(
    id="SawyerPushZoom-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushZoom",
    kwargs={},
    max_episode_steps=70
)

register(
    id="SawyerPushZoomEasy-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushZoomEasy",
    kwargs={},
    max_episode_steps=70
)

register(
    id="GymWalkerEasy-v0",
    entry_point="gym_mod.gymenvs:GymWalkerEasy",
    max_episode_steps=1000
)

register(
    id="GymInvertedPendulumEasy-v0",
    entry_point="gym_mod.gymenvs:GymInvertedPendulumEasy",
    max_episode_steps=1000
)

register(
    id="GymHalfCheetahEasy-v0",
    entry_point="gym_mod.gymenvs:GymHalfCheetahEasy",
    max_episode_steps=1000,
)

register(
    id="GymWalkerBackwards-v0",
    entry_point="gym_mod.gymenvs:GymWalkerBackwards",
    max_episode_steps=1000
)

register(
    id="GymWalkerWeight-v0",
    entry_point="gym_mod.gymenvs:GymWalkerWeight",
    max_episode_steps=1000,
)

register(
    id="GymWalkerColor-v0",
    entry_point="gym_mod.gymenvs:GymWalkerColor",
    max_episode_steps=1000,
)

register(
    id="GymWalkerDM-v0",
    entry_point="gym_mod.gymenvs:GymWalkerDM",
    max_episode_steps=1000,
)

register(
    id="GymWalkerColorNoBackground-v0",
    entry_point="gym_mod.gymenvs:GymWalkerColorNoBackground",
    max_episode_steps=1000
)

register(
    id="GymWalkerWeightColor-v0",
    entry_point="gym_mod.gymenvs:GymWalkerWeightColor",
    max_episode_steps=1000
)

register(
    id="GymWalker100-v0",
    entry_point="gym_mod.gymenvs:GymWalker",
    max_episode_steps=100
)

register(
    id="GymWalkerColor100-v0",
    entry_point="gym_mod.gymenvs:GymWalkerColor",
    max_episode_steps=100
)

register(
    id="GymInvertedPendulumWeight-v0",
    entry_point="gym_mod.gymenvs:GymInvertedPendulumWeight",
    max_episode_steps=1000,
)

register(
    id="GymInvertedPendulumWeightColor-v0",
    entry_point="gym_mod.gymenvs:GymInvertedPendulumWeightColor",
    max_episode_steps=1000
)

register(
    id="GymInvertedPendulumWeightColor2-v0",
    entry_point="gym_mod.gymenvs:GymInvertedPendulumWeightColor2",
    max_episode_steps=1000
)

register(
    id="GymInvertedPendulumDM-v0",
    entry_point="gym_mod.gymenvs:GymInvertedPendulumDM",
    max_episode_steps=1000,
)

register(
    id="GymInvertedPendulumWeightColor3-v0",
    entry_point="gym_mod.gymenvs:GymInvertedPendulumWeightColor3",
    max_episode_steps=1000
)

register(
    id="GymHalfCheetahWeight-v0",
    entry_point="gym_mod.gymenvs:GymHalfCheetahWeight",
    max_episode_steps=1000,
)

register(
    id="GymHalfCheetahDM-v0",
    entry_point="gym_mod.gymenvs:GymHalfCheetahDM",
    max_episode_steps=1000,
)

register(
    id="GymHalfCheetahArmature-v0",
    entry_point="gym_mod.gymenvs:GymHalfCheetahArmature",
    max_episode_steps=1000,
)

register(
    id="GymHopperWeight-v0",
    entry_point="gym_mod.gymenvs:GymHopperWeight",
    max_episode_steps=1000,
)

register(
    id="GymHopperColor-v0",
    entry_point="gym_mod.gymenvs:GymHopperColor",
    max_episode_steps=1000,
)

register(
    id="GymHopperColor2-v0",
    entry_point="gym_mod.gymenvs:GymHopperColor2",
    max_episode_steps=1000,
)

register(
    id="GymHopperWeightColor-v0",
    entry_point="gym_mod.gymenvs:GymHopperWeightColor",
    max_episode_steps=1000,
)

register(
    id="GymHalfCheetahColor-v0",
    entry_point="gym_mod.gymenvs:GymHalfCheetahColor",
    max_episode_steps=1000,
)

register(
    id="GymInvertedPendulumColor-v0",
    entry_point="gym_mod.gymenvs:GymInvertedPendulumColor",
    max_episode_steps=1000
)

register(
    id="GymInvertedPendulumBackground-v0",
    entry_point="gym_mod.gymenvs:GymInvertedPendulumBackground",
    max_episode_steps=1000
)

register(
    id="GymInvertedPendulumViewpoint-v0",
    entry_point="gym_mod.gymenvs:GymInvertedPendulumViewpoint",
    max_episode_steps=1000
)

register(
    id="GymReacher-v2",
    entry_point="gym_mod.gymenvs:GymReacher",
    max_episode_steps=50
)

register(
    id="GymReacherColor-v0",
    entry_point="gym_mod.gymenvs:GymReacherColor",
    max_episode_steps=50
)

register(
    id="S2RDartWalker-v0",
    entry_point="gym_mod.dart_envs:S2RDartWalker",
    max_episode_steps=1000,
)

register(
    id="InaccurateMinitaurTrottingEnv-v0",
    entry_point="gym_mod.bullet_env:InaccurateMinitaurTrottingEnv",
    max_episode_steps=1000,
)

register(
    id="AccurateMinitaurTrottingEnv-v0",
    entry_point="gym_mod.bullet_env:AccurateMinitaurTrottingEnv",
    max_episode_steps=1000,
)

register(
    id="SawyerPush-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushEnv",
    kwargs={},
    max_episode_steps=70
)

register(
    id="SawyerPushEasy-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushEnvEasy",
    kwargs={},
    max_episode_steps=70
)

register(
    id="SawyerPushColor-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushColor",
    kwargs={},
    max_episode_steps=70
)

register(
    id="SawyerPushBackground-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushBackground",
    kwargs={},
    max_episode_steps=70
)

register(
    id="SawyerPushColorShiftView-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushColorShiftView",
    kwargs={},
    max_episode_steps=70
)

register(
    id="SawyerPushShiftView-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushShiftView",
    kwargs={},
    max_episode_steps=70
)

register(
    id="SawyerPushShiftViewBackground-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushShiftViewBackground",
    kwargs={},
    max_episode_steps=70
)

register(
    id="SawyerPushRobotTopView-v0",
    entry_point="gym_mod.sawyer_env:SawyerPushRobotTopView",
    kwargs={},
    max_episode_steps=70
)

register(
    id="Sliding-v0",
    entry_point="gym_mod.robosuite_extra.slide_env:SawyerSlide",
    kwargs={},
    max_episode_steps=70
)

#suite._DOMAINS["s2rwalker"] = dm_walker_s2r

