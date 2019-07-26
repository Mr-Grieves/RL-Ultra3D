import sys
sys.path.append('/home/nathanvw/dev/RL/gym-ultra3d')
import gym_ultra3d
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ENV_NAME = 'Ultra3D-v2'
env3d = gym.make(ENV_NAME)

N = 50
x_arr = np.linspace(-1,1,N)

# Sweep through all thetas
if False:
    for t in x_arr:
        env3d.force_reset(t, 0, 0)
        [ob, reward, episode_over,j] = env3d.step(6)
        env3d.render()

# Sweep through all phis
if True:
    for p in x_arr:
        env3d.force_reset(0, p, 0.5)
        [ob, reward, episode_over,j] = env3d.step(6)
        env3d.render()

# Sweep through all distances
if False:
    for d in x_arr:
        env3d.force_reset(0, 0, d)
        [ob, reward, episode_over,j] = env3d.step(6)
        env3d.render()

