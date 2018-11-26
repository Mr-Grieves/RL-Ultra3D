import random

import gym
from gym import spaces
import numpy as np
import math
import scipy.ndimage.interpolation as spi
import scipy.misc as spm
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import logging
logger = logging.getLogger(__name__)

INFILE = '/home/nathanvw/dev/RL/data/3DUS/np/np06resized.npy'
MASKFILE = '/home/nathanvw/dev/RL/data/3DUS/np/mask.png'
ALPHA_MAX = 0.1

TARGET_D = 0.01
TARGET_A = 0.01
NETSIZE = 128

ACTION_LOOKUP = {
    0: [-1, 1],
    1: [-1, 0],
    2: [-1, 1],
    3: [0 ,-1],
    # Can't sit still!
    4: [0 , 1],
    5: [1 ,-1],
    6: [1 , 0],
    7: [1 , 1]
}

class Ultra3DEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__version__ = "0.0"
        logging.info("Ultra3D - Version {}".format(self.__version__))

        # Load the np-converted dataset
        self.data = np.load(INFILE)
        dims = self.data.shape
        self.x0 = dims[0]
        self.y0 = dims[1]
        self.z0 = dims[2]

        # Load the mask?
        # TODO: ...
        #self.mask = np.transpose(np.flipud(spm.imread(MASKFILE)[:,:,1]))
        #self.mask = np.uint8(self.mask/255)

        # Set the target image
        TrueAP4 = self.get_slice(TARGET_D, TARGET_A)
        ###self.display_slice(TrueAP4)
        self.TrueAP4_masked = self.mean_mask(TrueAP4)
        self.maxreward = self.correlate(TrueAP4)

        # Define what the agent can do
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))

        # Observation space is the range of valid states
        self.observation_space = spaces.Box(low=0., high=1., shape=(self.x0,self.y0))

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
        #print('BEFOR: curr_d =',self.curr_d,'\tcurr_a =',self.curr_a,'\talpha =',self.alpha)
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        if reward > 1.25:
            #print("Close enough! TrueAP4 Found!")
            #reward = 100
            episode_over = True
        elif reward < -0.4:
            #print("Too far, exiting")
            episode_over = True
        else:
            episode_over = False

        #print('curr_d =',self.curr_d,'\tcurr_a =',self.curr_a,'\treward =',reward,'\talpha =',self.alpha)
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        #self.action_episode_memory[self.curr_episode].append(action)
        a = ACTION_LOOKUP[action]
        self.curr_d = self.curr_d + self.alpha*next(iter(a))
        self.curr_a = self.curr_a + self.alpha*next(iter(a))

        # Threshold actions
        self.curr_d = min(1,max(-1,self.curr_d))
        self.curr_a = min(1,max(-1,self.curr_a))
        self.curr_slice = self.get_slice(self.curr_d, self.curr_a)

    def _get_reward(self):
        curr_slice_masked = self.mean_mask(self.curr_slice)
        x = (2*self.correlate(curr_slice_masked)-self.maxreward)/self.maxreward
        self.alpha = ALPHA_MAX #* (1-x)/2  # Adjust alpha based on current reward
        return x #math.exp(3*x-0.5)-1

    def _get_state(self):
        return spm.imresize(np.divide(self.curr_slice,255.,dtype=np.float),(NETSIZE,NETSIZE))

    def reset(self,d,a):
        #self.curr_episode += 1
        #self.action_episode_memory.append([])

        # Random init of state
        self.curr_d = d
        self.curr_a = a
        self.alpha = ALPHA_MAX
        self.curr_slice = self.get_slice(self.curr_d, self.curr_a)
        return self._get_state()

    def render(self, mode='human', close=False):
        self.display_slice(self.curr_slice)

    """ Parameters:
        data = 3D numpy array representing the raw 3DUS dataset
        state = the desired
         slice
            dist_n: the normalized distance of the desired plane from the origin
                -1 = lower bound
                0  = centre slice
                1  = upper bound
            angle_n: the normalized angle of the desired plane 
                -1 = -90 degress
                0  = no rotation
                1  = 90 degrees"""
    def get_slice(self, dist_n, angle_n):
        angle = angle_n*math.pi
        dist = dist_n*(self.x0-1)/2
        #print("Requested angle =", math.degrees(angle), "deg \tdist =", dist)

        # --- 1: Find x_i and y_i's ---
        # 1.0 Corner cases
        if angle==0 or angle==math.pi/2 or angle==math.pi:
            angle -= 0.0000001
        if angle==-math.pi/2 or angle==-math.pi:
            angle += 0.0000001

        # 1.1 first x1,x2 and y1,y2
        x1 = (self.x0 + self.y0*math.tan(angle))/2 + dist/math.cos(angle)
        x2 = (self.x0 - self.y0*math.tan(angle))/2 + dist/math.cos(angle)
        y1 = self.y0/2 - self.x0/math.tan(abs(angle))/2 + dist/math.sin(angle)
        y2 = self.y0/2 + self.x0/math.tan(abs(angle))/2 + dist/math.sin(angle)

        # 1.2 more corner cases
        if abs(angle) > math.pi/2:
            temp = x2
            x2 = x1
            x1 = temp

        # 1.3 Threshold
        x1,y1 = self.threshold(x1,y1)
        x2,y2 = self.threshold(x2,y2)
        #print("x1=",x1," x2=",x2,"\ty1=",y1," y2=",y2)

        # 1.4 Calc x_i's, y_i's
        len_px = round(math.sqrt(math.pow((y1-y2),2)+math.pow((x1-x2),2)))+1
        x_i = np.linspace(x1,x2,len_px)
        y_i = np.linspace(y1,y2,len_px)
        #print("length =",len_px)

        # --- 2: Construct slice from x/y_i's ---
        slice = np.zeros((len_px,self.z0),dtype='uint8')
        for i in range(0,len_px):
            x_c, y_c = int(round(x_i[i])), int(round(y_i[i]))
            slice[i,:] = self.data[x_c,y_c,:]

        # --- 3: Mask slice ---
        mid = len_px/2
        if(len_px > self.x0):
            start = math.floor(mid-self.x0/2)
            end = math.floor(mid+self.x0/2)
            #print('start =',start,'end =',end,)
            slice = slice[start:end,:]
        elif len_px < self.x0:
            leftpad = math.floor((self.x0-len_px)/2)
            rightpad = self.x0 - leftpad - len_px
            #print('left=',leftpad,'right=',rightpad, 'sum =',leftpad+rightpad+len_px)
            slice = np.concatenate((
                np.zeros((leftpad,self.z0),dtype='uint8'),
                slice,
                np.zeros((rightpad,self.z0),dtype='uint8')))
        return slice

    def get_slice_slow(self, dist_n, angle_n):
        if dist_n > 1. or dist_n < -1.:
            raise RuntimeError("given distance exceeds threshold")
        if angle_n > 1. or angle_n < -1.:
            raise RuntimeError("given angle exceeds threshold")

        # Rotate the volume
        angle = round(angle_n*90)
        rot_vol = spi.rotate(self.data, angle, reshape=False, prefilter=False)

        # Translate second
        dist = math.floor(rot_vol.shape[0]/2 + dist_n*(self.x0-1)/2)

        # Slice the middle rotated volume
        slice = rot_vol[dist,:,:]
        return slice


    ''' Parameters:
            slice: mean-subtracted image'''
    def correlate(self, slice_masked):
        if(slice_masked.shape != self.TrueAP4_masked.shape):
            print("Images aren't the same shape!")
            return 0
        xcorr2d = np.multiply(slice_masked,self.TrueAP4_masked)
        return sum(sum(xcorr2d))

    def mean_mask(self, image):
        #TODO: image = np.multiply(image,self.mask)
        true_mean = np.sum(np.sum(image,dtype='int32'),dtype='int32')/np.count_nonzero(image)
        #print("mean =",image.mean(),"\ttrue mean = ",true_mean)
        nonzeros = np.nonzero(image)
        nzs_x = nonzeros[0]
        nzs_y = nonzeros[1]
        #print("number of nonzeros =",nzs_x.size)
        masked = np.zeros(image.shape,dtype='float')
        for i in range(0,nzs_x.size):
            #print('x =',nzs_x[i],' y =',nzs_y[i])
            masked[nzs_x[i]][nzs_y[i]] = float(image[nzs_x[i]][nzs_y[i]]) - true_mean
        return masked

    def display_slice(self, slice):
        slice = np.flipud(np.transpose(slice))
        plt.imshow(slice, interpolation='nearest')
        plt.show()

    def seed(self, seed):
        random.seed(seed)

    def threshold(self, xi, yi):
        xo = max(0, min(self.x0 - 1, xi))
        yo = max(0, min(self.y0 - 1, yi))
        return xo, yo

N = 40
env3d = Ultra3DEnv()
d_arr = np.linspace(-1,1,N)
a_arr = np.linspace(-1,1,N)

x = np.zeros((N,N),dtype=float)
y = np.zeros((N,N),dtype=float)
z = np.zeros((N,N),dtype=float)
print(z.shape)

d_idx = 0
for d in d_arr:
    a_idx = 0
    for a in a_arr:
        print("d =",d," a =",a)
        env3d.reset(d,a)
        [ob, reward, episode_over,j] = env3d.step(4)
        x[d_idx,a_idx] = d
        y[d_idx,a_idx] = a
        z[d_idx,a_idx] = reward
        a_idx += 1
    d_idx += 1

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
Axes3D.plot_wireframe(ax,x,y,z)
fig.show()

input("Press something...")