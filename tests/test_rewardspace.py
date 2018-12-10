import sys
sys.path.append('/home/nathanvw/dev/RL/gym-ultra3d')
import gym_ultra3d
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

ENV_NAME = 'Ultra3D-v2'
env3d = gym.make(ENV_NAME)

d = -1
N = 20
th_arr = np.linspace(-.1,.1,N)
ph_arr = np.linspace(-1,1,N)

x = np.zeros((N,N),dtype=float)
y = np.zeros((N,N),dtype=float)
z = np.zeros((N,N),dtype=float)
print(z.shape)

t = time.time()
th_idx = 0
for th in th_arr:
    ph_idx = 0
    max_reward = -10
    for ph in ph_arr:
        env3d.force_reset(th,ph,d)
        env3d.render()
        [ob, reward, episode_over,j] = env3d.step(26)
        x[th_idx,ph_idx] = th
        y[th_idx,ph_idx] = ph
        z[th_idx,ph_idx] = reward
        ph_idx += 1
        if(reward > max_reward): max_reward = reward
        #print(reward)
    print("Max reward for th =", th,"line was =",max_reward)
    th_idx += 1
print("d =",d,"took",time.time()-t,"secs")

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
Axes3D.plot_wireframe(ax,x,y,z)
ax.plot(xs=[0.02,0.02], ys=[0.02,0.02], zs=[-1, 2], color='g')
fig.show()

input("Press something...")