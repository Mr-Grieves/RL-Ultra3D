import sys
sys.path.append('/home/nathanvw/dev/RL/gym-ultra3d')
import gym_ultra3d
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ENV_NAME = 'Ultra3D-v3'
env3d = gym.make(ENV_NAME)

N = 50
arr = np.linspace(-1,1,N)

for x in arr:
    env3d.force_reset(0,0,x,0)
    [ob, reward, episode_over,j] = env3d.step(8)
    env3d.render()

input("Press key to terminate...")

