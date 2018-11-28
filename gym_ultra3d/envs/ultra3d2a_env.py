import random

import gym
from gym import spaces
import numpy as np
import math
import scipy.ndimage.interpolation as spi
import scipy.misc as spm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image

import logging
logger = logging.getLogger(__name__)

INFILE = 'data/3DUS/np/np06resized.npy'
MASKFILE = 'data/3DUS/np/mask.png'
PHI_MAX = 30
ALPHA_MAX = 0.1
NETSIZE = 128

HIGH_REWARD_THRESH = 1.35
LOW_REWARD_THRESH = -0.43

NUM_STEPS_MAX = 100
TARGET_THETA = 0.02
TARGET_PHI = 0.02

class Ultra3DEnv2A(gym.Env):
    metadata = {'render.modes': ['human']}

    # Can't sit still!
    ACTION_LOOKUP = {
        0: [-1,-1],
        1: [-1, 0],
        2: [-1, 1],
        3: [ 0,-1],
        #: [ 0, 0],
        4: [ 0, 1],
        5: [ 1,-1],
        6: [ 1, 0],
        7: [ 1, 1]
    }

    def __init__(self):
        self.__version__ = "0.0"
        logging.info("Ultra3D - Version {}".format(self.__version__))

        # Load the np-converted dataset
        self.data = np.load(INFILE)
        self.data = np.flip(self.data,axis=2)
        dims = self.data.shape
        self.x0 = dims[0]
        self.y0 = dims[1]
        self.z0 = dims[2]

        # Load the mask?
        self.mask = spm.imread(MASKFILE)[:,:,1]
        self.mask = np.uint8(self.mask/255)

        # Set the target image
        TrueAP4 = self.get_slice(TARGET_THETA, TARGET_PHI)
        self.TrueAP4_masked = self.mean_mask(TrueAP4)
        self.maxreward = self.correlate(TrueAP4)
        self.display_slice(TrueAP4)

        # Define what the agent can do
        self.action_space = spaces.Discrete(len(self.ACTION_LOOKUP))

        # Observation space is the range of valid states
        self.observation_space = spaces.Box(low=0., high=1., shape=(self.x0,self.y0))

        # Set up plot in case visualize is on
        self.plot_opened = False
        self.success = 0
        self.num_steps = 0

        # Store what the agent tried
        #self.curr_episode = -1
        #self.action_episode_memory = []

    def step(self, action):
        """
        Parameters
        ----------
        action: From the ACTION_LOOKUP table. e.g. {0,-1}

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        #print('BEFOR: curr_th =',self.curr_th,'\tcurr_ph =',self.curr_ph,'\talpha =',self.alpha)
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        self.num_steps += 1
        if reward > HIGH_REWARD_THRESH:
            #print("Close enough! TrueAP4 Found!")
            episode_over = True
            reward = 100
            self.success = 1
        elif reward < LOW_REWARD_THRESH:
            #print("Too far, exiting")
            episode_over = True
            reward = -10
            self.success = -1
        elif self.num_steps >= NUM_STEPS_MAX:
            episode_over = True
            reward = -NUM_STEPS_MAX
            self.success = -1
        else:
            episode_over = False
            self.success = 0

        #print('Just took action #',action,': curr_th =',self.curr_th,'\tcurr_ph =',self.curr_ph,'\treward =',reward,'\talpha =',self.alpha)
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        #self.action_episode_memory[self.curr_episode].append(action)
        act = self.ACTION_LOOKUP[action]
        self.curr_th = self.curr_th + self.alpha*act[0]
        self.curr_ph = self.curr_ph + self.alpha*act[1]

        # Threshold actions
        self.curr_th = min(1,max(-1,self.curr_th))
        self.curr_ph = min(1,max(-1,self.curr_ph))
        self.curr_slice = self.get_slice(self.curr_th, self.curr_ph)

    def _get_reward(self):
        curr_slice_masked = self.mean_mask(self.curr_slice)
        x = (2*self.correlate(curr_slice_masked)-self.maxreward)/self.maxreward
        self.alpha = ALPHA_MAX*(1-x)/2  # Adjust alpha based on current reward
        return x+0.5 #math.exp(3*x-0.5)-1

    def _get_state(self):
        return np.array(spm.imresize(self.curr_slice,(NETSIZE,NETSIZE)),dtype='float') / 255.0

    def reset(self):
        #self.curr_episode += 1
        #self.action_episode_memory.append([])

        # Random init of state
        self.curr_th = random.uniform(-1,1)
        self.curr_ph = random.uniform(-1,1)
        self.alpha = ALPHA_MAX
        self.curr_slice = self.get_slice(self.curr_th, self.curr_ph)
        self.num_steps = 0
        return self._get_state()

    def force_reset(self,theta,phi):
        # Random init of state
        self.curr_th = theta
        self.curr_ph = phi
        self.alpha = ALPHA_MAX
        self.curr_slice = self.get_slice(self.curr_th, self.curr_ph)
        return self._get_state()

    def render(self, mode='human', close=False):
        #print("rendering with d =",self.curr_th," and a =",self.curr_ph)
        if self.plot_opened == False:
            self.plot_opened = True
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        # Plot wireframe
        colour_map = { -1 : 'r',
                        0 : 'k',
                        1 : 'g'}
        frame_colour = colour_map.get(self.success)
        self.ax.plot(xs=[-1, -1], ys=[-1, -1], zs=[-1, 1], color=frame_colour)
        self.ax.plot(xs=[-1, -1], ys=[-1, 1], zs=[-1, -1], color=frame_colour)
        self.ax.plot(xs=[-1, 1], ys=[-1, -1], zs=[-1, -1], color=frame_colour)
        self.ax.plot(xs=[1, 1], ys=[1, 1], zs=[-1, 1], color=frame_colour)
        self.ax.plot(xs=[1, 1], ys=[-1, 1], zs=[1, 1], color=frame_colour)
        self.ax.plot(xs=[-1, 1], ys=[1, 1], zs=[1, 1], color=frame_colour)
        self.ax.plot(xs=[-1, -1], ys=[1, 1], zs=[-1, 1], color=frame_colour)
        self.ax.plot(xs=[-1, -1], ys=[-1, 1], zs=[1, 1], color=frame_colour)
        self.ax.plot(xs=[-1, 1], ys=[-1, -1], zs=[1, 1], color=frame_colour)
        self.ax.plot(xs=[1, 1], ys=[-1, -1], zs=[-1, 1], color=frame_colour)
        self.ax.plot(xs=[1, 1], ys=[-1, 1], zs=[-1, -1], color=frame_colour)
        self.ax.plot(xs=[-1, 1], ys=[1, 1], zs=[-1, -1], color=frame_colour)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Plot target
        b1, b2, b3, b4 = self.get_bounding_box_full(TARGET_THETA, TARGET_PHI)
        X = [[b1[0], b2[0]], [b3[0], b4[0]]]
        Y = [[b1[1], b2[1]], [b3[1], b4[1]]]
        Z = [[b1[2], b2[2]], [b3[2], b4[2]]]
        self.ax.plot_surface(X, Y, Z, alpha=0.5)

        # Plot surface
        b1, b2, b3, b4 = self.get_bounding_box_full(self.curr_th, self.curr_ph)
        X = [[b1[0], b2[0]], [b3[0], b4[0]]]
        Y = [[b1[1], b2[1]], [b3[1], b4[1]]]
        Z = [[b1[2], b2[2]], [b3[2], b4[2]]]
        self.ax.plot_surface(X, Y, Z, alpha=0.5)
        if(self.success):
            self.ax.plot_surface(X, Y, Z, alpha=0.5, color=frame_colour)
        else:
            self.ax.plot_surface(X, Y, Z, alpha=0.5)

        plt.draw()
        plt.pause(0.0001)
        plt.cla()
        return

    def get_bounding_box(self, theta, phi):
        h1 = [self.x0 / 2 + self.y0 / 2 * math.sin(theta),
              self.y0 / 2 - self.y0 / 2 * math.cos(theta)]

        h2 = [self.x0 / 2 - self.y0 / 2 * math.sin(theta),
              self.y0 / 2 + self.y0 / 2 * math.cos(theta)]

        z_min = self.z0 / 2 - self.z0 / 2 * math.cos(phi)
        z_max = self.z0 / 2 + self.z0 / 2 * math.cos(phi)
        return h1, h2, z_min, z_max

    def get_bounding_box_full(self, theta_n, phi_n):
        theta = theta_n*math.pi
        phi = math.radians(phi_n*PHI_MAX)

        b1 = [ math.sin(theta) - math.sin(phi)*math.cos(theta),
              -math.cos(theta) - math.sin(phi)*math.sin(theta),
              -math.cos(phi)]

        b2 = [-math.sin(theta) - math.sin(phi)*math.cos(theta),
               math.cos(theta) - math.sin(phi)*math.sin(theta),
              -math.cos(phi)]

        b3 = [ math.sin(theta) + math.sin(phi)*math.cos(theta),
              -math.cos(theta) + math.sin(phi)*math.sin(theta),
               math.cos(phi)]

        b4 = [-math.sin(theta) + math.sin(phi)*math.cos(theta),
               math.cos(theta) + math.sin(phi)*math.sin(theta),
               math.cos(phi)]

        #print("theta =",math.degrees(theta),"\tphi =",math.degrees(phi),"\t\th1 =,",h1,"\th2 =",h2,"\tv1 =",v1,"\tv2 =",v2)
        return b1, b2, b3, b4


    def get_slice(self, theta_n, phi_n):
        theta = theta_n*math.pi
        phi = math.radians(phi_n*PHI_MAX)

        # --- 1: Get bounding box dims ---
        h1, h2, z_min, z_max = self.get_bounding_box(theta=theta,phi=phi)
        w = self.y0
        h = self.z0
        slice = np.zeros((self.y0,self.z0),dtype='uint8')
        z_i = np.linspace(z_min, z_max, h)

        # --- 2: Extract lines from volume ---
        for j in range(0, h-1):
            # Get layer at current z
            z_curr = int(round(z_i[j]))
            layer = self.data[:,:,z_curr]

            # Get x_i and y_i for current layer
            x_offset = (j-h/2) * math.sin(phi) * math.cos(theta)
            y_offset = (j-h/2) * math.sin(phi) * math.sin(theta)
            x_i = np.array(np.rint(np.linspace(h1[0],h2[0],w) + x_offset),dtype='int')
            y_i = np.array(np.rint(np.linspace(h1[1],h2[1],w) + y_offset),dtype='int')

            # Flatten
            flat_inds = np.ravel_multi_index((x_i,y_i),(w,h),mode='clip')

            # Fill in line
            slice[j,:] = np.take(layer,flat_inds)

        # --- 3: Mask slice ---
        slice = np.multiply(slice, self.mask)
        return slice

    ''' Parameters:
            slice: mean-subtracted image'''
    def correlate(self, slice_masked):
        if slice_masked.shape != self.TrueAP4_masked.shape:
            print("Images aren't the same shape!")
            return 0
        xcorr2d = np.multiply(slice_masked,self.TrueAP4_masked)
        return sum(sum(xcorr2d))

    def mean_mask(self, image):
        true_mean = np.sum(np.sum(image,dtype='int32'),dtype='int32')/np.count_nonzero(image)
        nonzeros = np.nonzero(image)
        nzs_x = nonzeros[0]
        nzs_y = nonzeros[1]
        masked = np.zeros(image.shape,dtype='float')
        for i in range(0,nzs_x.size):
            masked[nzs_x[i]][nzs_y[i]] = float(image[nzs_x[i]][nzs_y[i]]) - true_mean
        return masked

    def display_slice(self, slice):
        if(slice.dtype == 'float'):
            mi = np.amin(np.amin(slice))
            ma = np.amax(np.amax(slice))
            slice = np.uint8((slice+mi)/(ma-mi)*255)
        img = Image.fromarray(slice, 'L')
        img.show()

    def seed(self, seed):
        random.seed(seed)

