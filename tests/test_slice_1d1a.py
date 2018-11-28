import numpy as np
import math
import scipy.ndimage.interpolation as spi
import scipy.misc as spm
from PIL import Image
import time
import matplotlib.pyplot as plt

INFILE = '../data/3DUS/np/np06resized.npy'
MASKFILE = '../data/3DUS/np/mask.png'

class slice_test():
    def __init__(self):
        self.data = np.load(INFILE)
        print('type =',type(self.data),
              '\nshape =',self.data.shape,
              '\ndtype =',self.data.dtype)
        dims = self.data.shape
        self.x0 = dims[0]
        self.y0 = dims[1]
        self.z0 = dims[2]
        self.mask = np.transpose(np.flipud(spm.imread(MASKFILE)[:,:,1]))
        self.mask = np.uint8(self.mask/255)
        TrueAP4 = self.get_slice(0, 0)
        self.render(TrueAP4)
        self.TrueAP4_masked = self.mean_mask(TrueAP4)

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

    def threshold(self,xi,yi):
        xo = max(0,min(self.x0-1,xi))
        yo = max(0,min(self.y0-1,yi))
        return xo, yo

    def render(self, slice):
        slice = np.flipud(np.transpose(slice))
        plt.imshow(slice)
        plt.gray()
        plt.show()

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


st = slice_test()
N = 10
for i in range(-N,N+1):
    d = i/N/20
    a = i/N/20
    slice = st.get_slice(d,a)
    masked = st.mean_mask(slice)
    print("d =",d,"\ta =",a,"\t xcorr =",st.correlate(masked))