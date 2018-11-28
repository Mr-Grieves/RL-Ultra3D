import numpy as np
import math
import scipy.ndimage.interpolation as spi
import scipy.misc as spm
from PIL import Image
import time
import matplotlib.pyplot as plt

INFILE = '../data/3DUS/np/np06resized.npy'
MASKFILE = '../data/3DUS/np/mask.png'
MAX_PHI = 30

class slice_test():
    def __init__(self):
        self.data = np.load(INFILE)
        self.data = np.flip(self.data,axis=2)
        print('type =',type(self.data),
              '\nshape =',self.data.shape,
              '\ndtype =',self.data.dtype)

        dims = self.data.shape
        self.x0 = dims[0]
        self.y0 = dims[1]
        self.z0 = dims[2]

        self.mask = spm.imread(MASKFILE)[:,:,1]
        self.mask = np.uint8(self.mask/255)

        TrueAP4 = self.get_slice(0, 0)
        self.render(TrueAP4)
        self.TrueAP4_masked = self.mean_mask(TrueAP4)


    def get_bounding_box(self, theta, phi):
        h1 = [self.x0 / 2 + self.y0 / 2 * math.sin(theta),
              self.y0 / 2 - self.y0 / 2 * math.cos(theta)]
              #self.z0 / 2]

        h2 = [self.x0 / 2 - self.y0 / 2 * math.sin(theta),
              self.y0 / 2 + self.y0 / 2 * math.cos(theta)]
              #self.z0 / 2]

        #v1 = [self.x0 / 2 - self.z0 / 2 * math.sin(phi) * math.cos(theta),
        #      self.y0 / 2 - self.z0 / 2 * math.sin(phi) * math.sin(theta),
        z_min = self.z0 / 2 - self.z0 / 2 * math.cos(phi)

        #v2 = [self.x0 / 2 + self.z0 / 2 * math.sin(phi) * math.cos(theta),
        #      self.y0 / 2 + self.z0 / 2 * math.sin(phi) * math.sin(theta),
        z_max = self.z0 / 2 + self.z0 / 2 * math.cos(phi)

        #width = math.sqrt(math.pow(h1[0]-h2[0],2)+math.pow(h1[1]-h2[1],2)+math.pow(h1[2]-h2[2],2))
        #height = math.sqrt(math.pow(v1[0]-v2[0],2)+math.pow(v1[1]-v2[1],2)+math.pow(v1[2]-v2[2],2))
        #print("theta =",math.degrees(theta),"\tphi =",math.degrees(phi),"\t\th1 =,",h1,"\th2 =",h2,"\tv1 =",v1,"\tv2 =",v2)
        return h1, h2, z_min, z_max#v1, v2, int(round(width)), int(round(height))

    def get_slice(self, theta_n, phi_n):
        theta = theta_n*math.pi
        phi = math.radians(phi_n*MAX_PHI)

        h1, h2, z_min, z_max = self.get_bounding_box(theta=theta,phi=phi)
        w = self.y0 #assert(w == self.y0)
        h = self.z0 #assert(h == self.z0)
        slice = np.zeros((self.y0,self.z0),dtype='uint8')

        z_i = np.linspace(z_min, z_max, h)

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
            line = np.take(layer,flat_inds)
            slice[j,:] = line

        # --- 3: Mask slice ---
        slice = np.multiply(slice, self.mask)
        return slice

    def threshold(self,xi,yi):
        xo = max(0,min(self.x0-1,xi))
        yo = max(0,min(self.y0-1,yi))
        return xo, yo

    def render(self, slice):
        plt.imshow(slice)
        plt.gray()
        plt.show()

    def correlate(self, slice_masked):
        if slice_masked.shape != self.TrueAP4_masked.shape:
            print("Images aren't the same shape!")
            return 0
        xcorr2d = np.multiply(slice_masked, self.TrueAP4_masked)
        return sum(sum(xcorr2d))

    def mean_mask(self, image):
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


st = slice_test()


N = 10
for i in range(-N,N+1):
    t = i/N/20
    p = i/N/20
    slice = st.get_slice(t,p)
    masked = st.mean_mask(slice)
    print("theta =",t,"\tphi =",p,"\t xcorr =",st.correlate(masked))