import sys
sys.path.append('/home/nathanvw/dev/RL/gym-ultra3d')
import gym_ultra3d
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ENV_NAME = 'Ultra3D-v1'
env3d = gym.make(ENV_NAME)

N = 40
th_arr = np.linspace(-2,2,N)

th_idx = 0
for th in th_arr:
    env3d.force_reset(th,0)
    [ob, reward, episode_over,j] = env3d.step(4)
    env3d.render()
    th_idx += 1

