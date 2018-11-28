import numpy as np
import math
import scipy.ndimage.interpolation as spi
import scipy.misc as spm
from PIL import Image

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


    """ Parameters:
        data = 3D numpy array representing the raw 3DUS dataset
        state = the desired
         slice
            state[0] = the distance of the desired plane from the origin
                -1 
            state[1] = the angle of the desired plane
    """
    def get_slice(self, dist_n, angle_n):
        # Rotate the volume
        angle = round(angle_n*90)
        #print('\nRotating by',angle,'degs and shifting by',round(dist_n*self.x0/2),'pixels')
        rot_vol = spi.rotate(self.data, angle, reshape=False, prefilter=False)
        #print('Rot shape =',rot_vol.shape)

        # Translate second
        dist = round(rot_vol.shape[0]/2 + dist_n*self.x0/2)
        #print('Slicing at x =',dist)

        # Slice the middle rotated volume
        slice = rot_vol[dist,:,:]
        return slice

    def correlate(self, im1, im2):
        if(im1.size != im2.size):
            print("Images aren't the same shape!")
            return 0

        #mi1 = np.amin(np.amin(im1))
        #ma1 = np.amax(np.amax(im1))
        #print('min =',np.amin(np.amin(im1)),' max =',np.amax(np.amax(im1)))
        #im1uchar = np.array(im1,dtype='uint8')
        #print('min =',np.amin(np.amin(im1uchar)),' max =',np.amax(np.amax(im1uchar)))
        #self.render(im1uchar)

        #mi2 = np.amin(np.amin(im2))
        #ma2 = np.amax(np.amax(im2))
        #print('min =',np.amin(np.amin(im2)),' max =',np.amax(np.amax(im2)))
        #im2uchar = np.array(im2,dtype='uint8')
        #print('min =',np.amin(np.amin(im2uchar)),' max =',np.amax(np.amax(im2uchar)))
        #self.render(im2uchar)

        xcorr2 = np.multiply(im1,im2) #subtract(im1,im2) #sps.correlate2d(im1,im2,mode='same')
        #print(xcorr2.shape)

        return sum(sum(xcorr2))

    def get_slice_fast(self, dist_n, angle_n):
        # Rotate the volume
        a = math.radians(angle_n*90)    # requested rot in deg
        d = round(dist_n*self.x0/2)     # distance in pixels

        if(a):
            x1 = round(self.x0/2 - self.x0/2*math.tan(a) + d/math.cos(a))
            x2 = round(self.x0/2 + self.x0/2*math.tan(a) + d/math.cos(a))
            y1 = round(self.y0/2 - self.y0/2/math.tan(a) + d/math.sin(a))
            y2 = round(self.y0/2 + self.y0/2/math.tan(a) + d/math.sin(a))
            if(x1 == x2):x2=x2+1
            if(y1 == y2):y2=y2+1
            x1 = max(min(x1,x2),0)
            x2 = min(max(x1,x2),self.x0-1)
            y1 = max(min(y1,y2),0)
            y2 = min(max(y1,y2),self.y0-1)

            print('\nRotating by',angle_n*90,'degs and shifting by',d,'pixels')
            print('New bounds:\n\tx1 =',min(x1,x2),'x2 =',max(x1,x2),'\n\ty1 =',min(y1,y2),'y2 =',max(y1,y2))
            print('Before Rot shape =', self.data[min(x1,x2):max(x1,x2),min(y1,y2):max(y1,y2),:].shape)
            rot_vol = spi.rotate(self.data[min(x1,x2):max(x1,x2),min(y1,y2):max(y1,y2),:], angle_n*90, axes=(0,1), reshape=True,mode='nearest',prefilter=False)
            print('After Rot shape =',rot_vol.shape)

            # Always just take the middle slice after rotation
            slice_ind = math.floor(rot_vol.shape[0]/2)

        else:
            # Translate second
            rot_vol = self.data
            slice_ind = round(rot_vol.shape[0]/2 + dist_n*self.x0/2)

        print('Slicing at x =',slice_ind)
        slice = rot_vol[slice_ind,:,:]
        return slice

    def render(self, slice):
        slice = np.flipud(np.transpose(slice))
        if(slice.dtype == 'float'):
            mi = np.amin(np.amin(slice))
            ma = np.amax(np.amax(slice))
            slice = np.uint8((slice+mi)/(ma-mi)*255)
        img = Image.fromarray(slice, 'L')
        img.show()

    def mean_mask(self, image):
        #image = np.multiply(image,self.mask)
        true_mean = np.sum(np.sum(image,dtype='int32'),dtype='int32')/np.count_nonzero(image)
        #print("mean =",image.mean(),"\ttrue mean = ",true_mean)
        nonzeros = np.nonzero(image)
        nzs_x = nonzeros[0]
        nzs_y = nonzeros[1]
        print("number of nonzeros =",nzs_x.size)
        masked = np.zeros(image.shape,dtype='float')
        for i in range(0,nzs_x.size):
            #print('x =',nzs_x[i],' y =',nzs_y[i])
            masked[nzs_x[i]][nzs_y[i]] = float(image[nzs_x[i]][nzs_y[i]]) - true_mean
        return masked

ACTION_LOOKUP = {
    0 : {-1,-1},
    1 : {-1, 0},
    2 : {-1, 1},
    3 : {0 ,-1},
    4 : {0 , 0},
    5 : {0 , 1},
    6 : {1 ,-1},
    7 : {1 , 0},
    8 : {1 , 1},
}

x = {0.12,0.13}
x.add(x,ACTION_LOOKUP[3])
print(y)

st = slice_test()
N = 5 # number of xcorrs = 4^2 36
TrueAP4 = st.get_slice(0.05,0.05)
st.render(TrueAP4)
TrueAP4 = st.mean_mask(TrueAP4)

maxreward = st.correlate(TrueAP4,TrueAP4)

'''
FakeAP4 = st.get_slice(0.025,0.025)
st.render(FakeAP4)
FakeAP4 = st.mean_mask(FakeAP4)
print(st.correlate(TrueAP4,FakeAP4))'''

for i in range(-2,2):
    for j in range(-2,2):
        slice = st.get_slice(i/N, j/N)
        #st.render(slice,close=True)
        slice = st.mean_mask(slice)
        reward = (2*st.correlate(TrueAP4,slice)-maxreward)/maxreward
        print("xcorr for",i/N,"and",j/N,"  \t=",reward)

#st.render(st.get_slice_fast(alpha*0.5, alpha*0.4))
