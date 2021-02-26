from method.environments import make_env
from method.config import argparser
import environments
import time
import imageio
import numpy as np


config, unparsed = argparser()

env1 = make_env("S2RDartWalker-v0", config)
env2 = make_env("Walker2d-v2", config)

vid1, vid2, vid3 = [], [], []

env1.reset()
for _ in range(10):
    ob, reward, done, info = env1.step(env1.action_space.sample())
    img = env1.render(mode="rgb_array")
    if config.encoder_type == "cnn":
        vid1.append(ob['ob'][:3].transpose(1, 2, 0))
    else:
        vid1.append(img)
if len(vid1) > 0:
    imageio.mimsave('./vid1.gif', np.array(vid1))

env2.reset()
for _ in range(10):
    ob, reward, done, info = env2.step(env2.action_space.sample())
    img = env2.render(mode="rgb_array")
    if config.encoder_type == "cnn":
        vid2.append(ob['ob'][:3].transpose(1, 2, 0))
    else:
        vid2.append(img)
if len(vid2) > 0:
    imageio.mimsave('./vid2.gif', np.array(vid2))

env1.reset()
for _ in range(10):
    ob, reward, done, info = env1.step(env1.action_space.sample())
    env1.render(mode="rgb_array")
    if config.encoder_type == "cnn":
        vid3.append(ob['ob'][:3].transpose(1, 2, 0))
env1.close()

if len(vid3) > 0:
    imageio.mimsave('./vid3.gif', np.array(vid3))
