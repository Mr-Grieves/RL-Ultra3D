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

INFILE = 'data/3DUS/np/np06resized.npy'
MASKFILE = 'data/3DUS/np/mask.png'
PHI_MAX = 30
ALPHA_MAX = 0.1
NETSIZE = 128

HIGH_REWARD_THRESH = 1.37
LOW_REWARD_THRESH = -0.48

NUM_STEPS_MAX = 100
TARGET_THETA = 0.02
TARGET_PHI = 0.02
TARGET_D = -0.01

#WARMUP_INTERC = 0.2
#WARMUP_PORTION = 0.1

class Ultra3DEnv2A1D(gym.Env):
    metadata = {'render.modes': ['human']}

    '''ACTION_LOOKUP = {
        0: [-1,-1,-1],
        1: [-1,-1, 0],
        2: [-1,-1, 1],
        3: [-1, 0,-1],
        4: [-1, 0, 0],
        5: [-1, 0, 1],
        6: [-1, 1,-1],
        7: [-1, 1, 0],
        8: [-1, 1, 1],

        9:  [ 0,-1,-1],
        10: [ 0,-1, 0],
        11: [ 0,-1, 1],
        12: [ 0, 0,-1],
        # : [ 0, 0, 0]
        13: [ 0, 0, 1],
        14: [ 0, 1,-1],
        15: [ 0, 1, 0],
        16: [ 0, 1, 1],

        17: [ 1,-1,-1],
        18: [ 1,-1, 0],
        19: [ 1,-1, 1],
        20: [ 1, 0,-1],
        21: [ 1, 0, 0],
        22: [ 1, 0, 1],
        23: [ 1, 1,-1],
        24: [ 1, 1, 0],
        25: [ 1, 1, 1],
        #26: [ 0, 0, 0], # Can't sit still!
    }'''

    ACTION_LOOKUP = {
        0: [-1, 0,  0],
        1: [0, -1,  0],
        2: [0,  0, -1],
        3: [0,  0,  1],
        4: [0,  1,  0],
        5: [1,  0,  0],
    }

    def __init__(self):
        self.__version__ = "1.0"

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
        TrueAP4 = self.get_slice(TARGET_THETA, TARGET_PHI, TARGET_D)
        self.TrueAP4_masked = self.mean_mask(TrueAP4)
        self.maxreward = self.correlate(TrueAP4)
        #self.display_slice(TrueAP4)

        # Define what the agent can do
        self.action_space = spaces.Discrete(len(self.ACTION_LOOKUP))

        # Observation space is the range of valid states
        self.observation_space = spaces.Box(low=0., high=1., shape=(self.x0,self.y0))

        # Set up plot in case visualize is on
        self.plot_opened = False
        self.success = 0
        self.num_steps = 0

        # Store what the agent tried
        self.outcomes = np.zeros((4,1))
        #self.total_num_steps = 0
        #self.max_num_steps = -1
        #self.curr_episode = -1
        #self.action_episode_memory = []

    def set_verbosity(self, v):
        self.verbosity = v

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
            episode_over = True
            reward = 100
            self.success = 1
            self.outcomes[0] += 1
        elif reward < LOW_REWARD_THRESH:
            episode_over = True
            reward = -10
            self.success = -1
            self.outcomes[1] += 1
        elif self.num_steps >= NUM_STEPS_MAX:
            episode_over = True
            reward = -10
            self.success = -1
            self.outcomes[2] += 1
        elif self.oob:
            episode_over = True
            reward = -10
            self.success = -1
            self.outcomes[3] += 1
        else:
            episode_over = False
            self.success = 0

        if(self.verbosity): print('Just took action #',action,': curr_th =',self.curr_th,'\tcurr_ph =',self.curr_ph,'\treward =',reward,'\talpha =',self.alpha)
        return ob, reward, episode_over, {}


    def _take_action(self, action):
        #self.action_episode_memory[self.curr_episode].append(action)
        act = self.ACTION_LOOKUP[action]

        # Update theta
        self.curr_th = self.curr_th + self.alpha*act[0]
        self.curr_th = (self.curr_th+1)%2-1         # Wrap theta

        # Check if agent wants to go out of bounds
        if (self.curr_ph == -1. and act[1] == -1) or \
            (self.curr_ph == 1. and act[1] ==  1) or \
            (self.curr_d == -1. and act[2] == -1) or \
            (self.curr_d ==  1. and act[2] ==  1):
            self.oob = True

        # Update phi
        self.curr_ph = self.curr_ph + self.alpha*act[1]
        self.curr_ph = min(1,max(-1,self.curr_ph))  # Threshold phi

        # Update d
        self.curr_d = self.curr_d + self.alpha*act[2]
        self.curr_d = min(1,max(-1,self.curr_d)) # Threshold d

        self.curr_slice = self.get_slice(self.curr_th, self.curr_ph, self.curr_d)

    def _get_reward(self):
        curr_slice_masked = self.mean_mask(self.curr_slice)
        x = (2*self.correlate(curr_slice_masked)-self.maxreward)/self.maxreward
        self.alpha = ALPHA_MAX*(1-x)/2  # Adjust alpha based on current reward
        return x+0.5 #math.exp(3*x-0.5)-1

    # As far as the agent is concerned, the current state is the image we get by slicing the volume at the current p
    def _get_state(self):
        return np.array(spm.imresize(self.curr_slice,(NETSIZE,NETSIZE)),dtype='float') / 255.0

    def reset(self):
        #assert(self.max_num_steps != -1)
        #self.total_num_steps += self.num_steps

        # Pseudo-random init of state
        #rand_range = min(1.0, WARMUP_INTERC+(self.total_num_steps/self.max_num_steps*(1-WARMUP_INTERC)/WARMUP_PORTION))
        #print("Episode =",self.total_num_steps," ->  max =",rand_range)
        #self.curr_th = random.uniform(-rand_range, rand_range)
        #self.curr_ph = random.uniform(-rand_range, rand_range)
        #self.curr_d  = random.uniform(-rand_range, rand_range)

        # Random init of state
        self.curr_th = random.uniform(-1, 1)
        self.curr_ph = random.uniform(-1, 1)
        self.curr_d = random.uniform(-1, 1)

        self.alpha = ALPHA_MAX
        self.curr_slice = self.get_slice(self.curr_th, self.curr_ph, self.curr_d)
        self.num_steps = 0
        self.oob = False
        return self._get_state()

    #def set_maximum_steps(self, nb):
        #self.max_num_steps = nb

    def force_reset(self,theta,phi,dist):
        # Random init of state
        self.curr_th = theta
        self.curr_ph = phi
        self.curr_d  = dist
        self.alpha = ALPHA_MAX
        self.curr_slice = self.get_slice(self.curr_th, self.curr_ph, self.curr_d)
        self.num_steps = 0
        self.oob = False
        return self._get_state()

    def get_bounding_box(self, theta, phi, dist):
        h1 = [self.x0 / 2 + self.y0 / 2 * math.sin(theta) + dist*math.cos(theta),
              self.y0 / 2 - self.y0 / 2 * math.cos(theta) + dist*math.sin(theta)]

        h2 = [self.x0 / 2 - self.y0 / 2 * math.sin(theta) + dist*math.cos(theta),
              self.y0 / 2 + self.y0 / 2 * math.cos(theta) + dist*math.sin(theta)]

        z_min = self.z0 / 2 - self.z0 / 2 * math.cos(phi)
        z_max = self.z0 / 2 + self.z0 / 2 * math.cos(phi)
        return h1, h2, z_min, z_max

    def get_slice(self, theta_n, phi_n, dist_n):
        theta = theta_n*math.pi
        phi = math.radians(phi_n*PHI_MAX)
        dist = dist_n*self.x0/2 # +/- 104 pixels

        # --- 1: Get bounding box dims ---
        h1, h2, z_min, z_max = self.get_bounding_box(theta=theta, phi=phi, dist=dist)
        w = self.y0
        h = self.z0

        # --- 2: Extract slice from volume ---
        # Get x_i and y_i for current layer
        x_offsets = np.linspace(-h/2, h/2, h) * math.sin(phi) * math.cos(theta)
        y_offsets = np.linspace(-h/2, h/2, h) * math.sin(phi) * math.sin(theta)

        # Tile and transpose
        x_offsets = np.transpose(np.tile(x_offsets,(w,1)))
        y_offsets = np.transpose(np.tile(y_offsets,(w,1)))

        x_i = np.tile(np.linspace(h1[0], h2[0], w),(h,1))
        y_i = np.tile(np.linspace(h1[1], h2[1], w),(h,1))

        x_i = np.array(np.rint(x_i + x_offsets),dtype='int')
        y_i = np.array(np.rint(y_i + y_offsets),dtype='int')

        # Don't forget to include the index offset from z!
        z_i = np.transpose(np.tile(np.linspace(z_min, z_max, h),(w,1)))
        z_i = np.array(np.rint(z_i),dtype='int')

        # Flatten
        flat_inds = np.ravel_multi_index((x_i,y_i,z_i),(self.x0,self.y0,self.z0),mode='clip')

        # Fill in entire slice at once
        slice = np.take(self.data, flat_inds)

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

    def print_outcomes(self):
        print("Outcomes:\n"
              "\t Above T_max:   ",self.outcomes[0],'\n',
              "\t Below T_min:   ",self.outcomes[1],'\n',
              "\t Too many steps:",self.outcomes[2],'\n',
              "\t Out of Bounds: ",self.outcomes[3],'\n')

    # TODO: this does not adequately display out-of-bounds stuff
    def get_bounding_box_full(self, theta_n, phi_n, dist_n):
        theta = theta_n*math.pi
        phi = math.radians(phi_n*PHI_MAX)

        b1 = [math.sin(theta) - math.sin(phi)*math.cos(theta) + dist_n*math.cos(theta),
              -math.cos(theta) - math.sin(phi)*math.sin(theta) + dist_n*math.sin(theta),
              -math.cos(phi)]

        b2 = [-math.sin(theta) - math.sin(phi)*math.cos(theta) + dist_n*math.cos(theta),
              math.cos(theta) - math.sin(phi)*math.sin(theta) + dist_n*math.sin(theta),
              -math.cos(phi)]

        b3 = [math.sin(theta) + math.sin(phi)*math.cos(theta) + dist_n*math.cos(theta),
              -math.cos(theta) + math.sin(phi)*math.sin(theta) + dist_n*math.sin(theta),
              math.cos(phi)]

        b4 = [-math.sin(theta) + math.sin(phi)*math.cos(theta) + dist_n*math.cos(theta),
              math.cos(theta) + math.sin(phi)*math.sin(theta) + dist_n*math.sin(theta),
              math.cos(phi)]

        #print("theta =",math.degrees(theta),"\tphi =",math.degrees(phi),"\t\th1 =,",h1,"\th2 =",h2,"\tv1 =",v1,"\tv2 =",v2)
        return b1, b2, b3, b4

    def render(self, mode='human', close=False):
        #print("rendering with d =",self.curr_th," and a =",self.curr_ph)
        if self.plot_opened == False:
            self.plot_opened = True
            self.fig = plt.figure(figsize=(6,10))
            self.ax1 = self.fig.add_subplot(211, projection='3d')
            self.ax1.invert_zaxis()
            self.ax1.invert_yaxis()
            self.ax2 = self.fig.add_subplot(212)
            plt.ion()
            plt.show()

        # Plot wireframe
        colour_map = { -1 : 'r',
                        0 : 'k',
                        1 : 'g'}
        frame_colour = colour_map.get(self.success)
        self.ax1.plot(xs=[-1, -1], ys=[-1, -1], zs=[-1, 1], color=frame_colour)
        self.ax1.plot(xs=[-1, -1], ys=[-1, 1], zs=[-1, -1], color=frame_colour)
        self.ax1.plot(xs=[-1, 1], ys=[-1, -1], zs=[-1, -1], color=frame_colour)
        self.ax1.plot(xs=[1, 1], ys=[1, 1], zs=[-1, 1], color=frame_colour)
        self.ax1.plot(xs=[1, 1], ys=[-1, 1], zs=[1, 1], color=frame_colour)
        self.ax1.plot(xs=[-1, 1], ys=[1, 1], zs=[1, 1], color=frame_colour)
        self.ax1.plot(xs=[-1, -1], ys=[1, 1], zs=[-1, 1], color=frame_colour)
        self.ax1.plot(xs=[-1, -1], ys=[-1, 1], zs=[1, 1], color=frame_colour)
        self.ax1.plot(xs=[-1, 1], ys=[-1, -1], zs=[1, 1], color=frame_colour)
        self.ax1.plot(xs=[1, 1], ys=[-1, -1], zs=[-1, 1], color=frame_colour)
        self.ax1.plot(xs=[1, 1], ys=[-1, 1], zs=[-1, -1], color=frame_colour)
        self.ax1.plot(xs=[-1, 1], ys=[1, 1], zs=[-1, -1], color=frame_colour)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')

        # Plot target
        b1, b2, b3, b4 = self.get_bounding_box_full(TARGET_THETA, TARGET_PHI, TARGET_D)
        X = [[b1[0], b2[0]], [b3[0], b4[0]]]
        Y = [[b1[1], b2[1]], [b3[1], b4[1]]]
        Z = [[b1[2], b2[2]], [b3[2], b4[2]]]
        self.ax1.plot_surface(X, Y, Z, alpha=0.5)

        # Plot surface
        b1, b2, b3, b4 = self.get_bounding_box_full(self.curr_th, self.curr_ph, self.curr_d)
        X = [[b1[0], b2[0]], [b3[0], b4[0]]]
        Y = [[b1[1], b2[1]], [b3[1], b4[1]]]
        Z = [[b1[2], b2[2]], [b3[2], b4[2]]]
        self.ax1.plot_surface(X, Y, Z, alpha=0.5)
        if(self.success):
            self.ax1.plot_surface(X, Y, Z, alpha=0.5, color=frame_colour)
        else:
            self.ax1.plot_surface(X, Y, Z, alpha=0.5)

        # Display current slice
        self.ax2.imshow(self.curr_slice,cmap="gray")

        plt.draw()
        plt.pause(0.00001)
        plt.cla()
        self.ax1.clear()
        self.ax2.clear()
        return
