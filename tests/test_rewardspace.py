import sys
sys.path.append('..')
import gym_ultra3d
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ENV_NAME = 'Ultra3D-v1'
env3d = gym.make(ENV_NAME)

N = 4
th_arr = np.linspace(-1,1,N)
ph_arr = np.linspace(-1,1,N)

x = np.zeros((N,N),dtype=float)
y = np.zeros((N,N),dtype=float)
z = np.zeros((N,N),dtype=float)
print(z.shape)

th_idx = 0
for th in th_arr:
    ph_idx = 0
    print("th =", th)
    for ph in ph_arr:
        env3d.force_reset(th,ph)
        env3d.render()
        [ob, reward, episode_over,j] = env3d.step(4)
        x[th_idx,ph_idx] = th
        y[th_idx,ph_idx] = ph
        z[th_idx,ph_idx] = reward
        ph_idx += 1
        print(reward)
    th_idx += 1

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
Axes3D.plot_wireframe(ax,x,y,z)
fig.show()

input("Press something...")