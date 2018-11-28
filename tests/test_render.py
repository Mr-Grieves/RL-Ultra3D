import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

TARGET_D = 0.02
TARGET_A = 0.02

class doit():
    def __init__(self):
        self.x0 = 208
        self.y0 = 208

    def get_slice(self, dist_n, angle_n):
        angle = angle_n * math.pi
        dist = dist_n * (self.x0 - 1) / 2

        # --- 1: Find x_i and y_i's ---
        # 1.0 Corner cases
        if angle == 0 or angle == math.pi / 2 or angle == math.pi:
            angle -= 0.0000001
        if angle == -math.pi / 2 or angle == -math.pi:
            angle += 0.0000001

        # 1.1 first x1,x2 and y1,y2
        x1 = (self.x0 + self.y0 * math.tan(angle)) / 2 + dist / math.cos(angle)
        x2 = (self.x0 - self.y0 * math.tan(angle)) / 2 + dist / math.cos(angle)
        y1 = self.y0 / 2 - self.x0 / math.tan(abs(angle)) / 2 + dist / math.sin(angle)
        y2 = self.y0 / 2 + self.x0 / math.tan(abs(angle)) / 2 + dist / math.sin(angle)

        # 1.2 more corner cases
        if abs(angle) > math.pi / 2:
            temp = x2
            x2 = x1
            x1 = temp

        # 1.3 Threshold
        x1, y1 = self.threshold(x1, y1)
        x2, y2 = self.threshold(x2, y2)
        x1 = 2*x1 / self.x0 - 1
        x2 = 2*x2 / self.x0 - 1
        y1 = 2*y1 / self.y0 - 1
        y2 = 2*y2 / self.y0 - 1

        print("x1=",x1," x2=",x2,"\ty1=",y1," y2=",y2)
        return x1, x2, y1, y2

    def threshold(self,xi,yi):
        xo = max(0,min(self.x0-1,xi))
        yo = max(0,min(self.y0-1,yi))
        return xo, yo

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

doit = doit()
plt.ion()
plt.show()

N = 15
d_arr = np.linspace(-1,1,N)
a_arr = np.linspace(-1,1,N)
for d in d_arr:
    for a in a_arr:

        # Plot wireframe
        ax.plot(xs=[-1, -1], ys=[-1, -1], zs=[-1, 1], color='b')
        ax.plot(xs=[-1, -1], ys=[-1, 1], zs=[-1, -1], color='b')
        ax.plot(xs=[-1, 1], ys=[-1, -1], zs=[-1, -1], color='b')
        ax.plot(xs=[1, 1], ys=[1, 1], zs=[-1, 1], color='b')
        ax.plot(xs=[1, 1], ys=[-1, 1], zs=[1, 1], color='b')
        ax.plot(xs=[-1, 1], ys=[1, 1], zs=[1, 1], color='b')
        ax.plot(xs=[-1, -1], ys=[1, 1], zs=[-1, 1], color='b')
        ax.plot(xs=[-1, -1], ys=[-1, 1], zs=[1, 1], color='b')
        ax.plot(xs=[-1, 1], ys=[-1, -1], zs=[1, 1], color='b')
        ax.plot(xs=[1, 1], ys=[-1, -1], zs=[-1, 1], color='b')
        ax.plot(xs=[1, 1], ys=[-1, 1], zs=[-1, -1], color='b')
        ax.plot(xs=[-1, 1], ys=[1, 1], zs=[-1, -1], color='b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Plot target
        x1,x2,y1,y2 = doit.get_slice(TARGET_D,TARGET_A)
        X = [[x1,x2],[x1,x2]]
        Y = [[y1,y2],[y1,y2]]
        Z = [[-1,-1],[ 1, 1]]
        ax.plot_surface(X,Y,Z,alpha=0.5)

        # Plot surface
        x1,x2,y1,y2 = doit.get_slice(d,a)
        X = [[x1,x2],[x1,x2]]
        Y = [[y1,y2],[y1,y2]]
        Z = [[-1,-1],[ 1, 1]]
        ax.plot_surface(X,Y,Z,alpha=0.5)

        plt.draw()
        plt.pause(0.000001)
        plt.cla()
