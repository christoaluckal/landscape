import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import ros_numpy
import open3d as o3d
import scipy.ndimage as ndimage
# from scipy.ndimage import gaussian_filter1d
from numpy import convolve
import pickle

def smoothZ(z):
    
    # row smoothing
    for i in range(len(z)):
        z[i] = ndimage.gaussian_filter(z[i], sigma=3)

    # column smoothing
    for i in range(len(z[0])):
        z[:,i] = ndimage.gaussian_filter(z[:,i], sigma=3)

    return z

def convolve_1d(signal, kernel):
    kernel = kernel[::-1]
    return [
        np.dot(
            signal[max(0,i):min(i+len(kernel),len(signal))],
            kernel[max(-i,0):len(signal)-i*(len(signal)-len(kernel)<i)],
        )
        for i in range(1-len(kernel),len(signal))
    ]

def gradient2D(landscape,count):
        xs = landscape[:,0].reshape(count,count)
        ys = landscape[:,1].reshape(count,count)
        zs = landscape[:,2].reshape(count,count)

        z_mask = abs(zs) < 0.1

        # set masked values to 0 in zs
        zs[z_mask] = 0

        new_landscape = np.hstack((xs.flatten()[:,np.newaxis],ys.flatten()[:,np.newaxis],zs.flatten()[:,np.newaxis]))
    
        idx = np.arange(len(landscape))
        plt.scatter(xs.flatten(),zs.flatten(),alpha=0.5,label="XZ",s=1)
        plt.scatter(ys.flatten(),zs.flatten(),alpha=0.5,label="YZ",s=1)
        plt.legend()
        plt.savefig("cart_v_Z.png")

        del_z = []

        for i in range(len(zs)):
            gx = np.diff(zs[i]+1e-5)/(np.diff(xs[i])+1e-5)
            gy = np.diff(zs[i]+1e-5)/(np.diff(ys[i])+1e-5)

            gz = np.sqrt(gx**2 + gy**2)
            # gz = convolve_1d(gz, np.array([1,2,1])/4)
            # gz = ndimage.gaussian_filter(gz, sigma=3)
            gz = np.clip(gz,0,100)
            del_z.append(gz)

        plt.close()
        
        im_delz = np.copy(del_z)
        # im_delz = (im_delz - im_delz.min()) / (im_delz.max() - im_delz.min())
        # im_delz = 255 * im_delz
        plt.matshow(im_delz,cmap='viridis')
        plt.colorbar()
        plt.savefig("gradient.png")
        plt.close()

        del_z = np.array(del_z)
        del_z = np.round(del_z,6)
        
        del_z = ndimage.gaussian_filter(del_z, sigma=3)
        del_z = abs(del_z)
        
        # convert to grayscale
        del_z = (del_z - del_z.min()) / (del_z.max() - del_z.min())
        del_z = 255 * del_z

        del_z_inv = np.flip(del_z,0)
        plt.imshow(del_z_inv,cmap='gray')

        print("Landscape shape: ",landscape.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_landscape)
        o3d.visualization.draw_geometries([pcd])
        plt.imsave("landscape.png",del_z_inv,cmap='gray')

        plt.close()
        pass

class PPMMaker:
    def __init__(self):
        self.sub = rospy.Subscriber("/ouster_points", PointCloud2, self.ouster_cb)
        self.points = None
        self.occ_pub = rospy.Publisher("/landscape", OccupancyGrid, queue_size=1)

    def filter_distance(self,landscape,distance=5):
        # convert to polar
        r = np.sqrt(landscape[:,0]**2 + landscape[:,1]**2 + landscape[:,2]**2)
        r_mask = r < distance
        landscape = landscape[r_mask]

        return landscape

    def ouster_cb(self,msg):
        self.points:PointCloud2 = msg

        # convert to numpy array
        landscape = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.points)
        landscape = np.array(landscape)
        
        with open("landscape.pkl","wb") as f:
            pickle.dump(landscape,f)

        # remove all z values greater than 1
        landscape = landscape[landscape[:,2] < 1]

        # get median z value
        min_z = np.median(landscape[:,2])
        landscape[:,2] = landscape[:,2] - min_z

        # filter distance
        landscape = self.filter_distance(landscape,distance=5.5)

        landscape = landscape[::5]

        n = int(np.sqrt(len(landscape)))

        landscape = landscape[:n**2]

        ind = np.lexsort((landscape[:,1],landscape[:,0]))

        landscape = landscape[ind]

        print("X min: ",landscape[:,0].min())
        print("X max: ",landscape[:,0].max())
        print("Y min: ",landscape[:,1].min())
        print("Y max: ",landscape[:,1].max())

        gradient2D(landscape,n)
        

if __name__ == "__main__":
    rospy.init_node("gauss")
    p = PPMMaker()
    rospy.spin()