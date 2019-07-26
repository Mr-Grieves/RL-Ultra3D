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
t_arr = np.linspace(-1,1,N)
p_arr = np.linspace(-1,1,N)
d_arr = np.linspace(-1,1,N)

for t in t_arr:
    env3d.force_reset(0.1, t, 0)
    [ob, reward, episode_over,j] = env3d.step(6)
    env3d.render()

