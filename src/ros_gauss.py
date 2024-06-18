import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import ros_numpy
import open3d as o3d
import scipy.ndimage as ndimage

def smoothZ(z):
    
    # row smoothing
    for i in range(len(z)):
        z[i] = ndimage.gaussian_filter(z[i], sigma=3)

    # column smoothing
    for i in range(len(z[0])):
        z[:,i] = ndimage.gaussian_filter(z[:,i], sigma=3)

    return z


def gradient2D(landscape,count):
        xs = landscape[:,0].reshape(count,count)
        ys = landscape[:,1].reshape(count,count)
        zs = landscape[:,2].reshape(count,count)
    
        idx = np.arange(len(landscape))

        # plt.plot(idx,landscape[:,0],alpha=0.5,label="X")
        # plt.plot(idx,landscape[:,1],alpha=0.5,label="Y")
        # plt.plot(idx,landscape[:,2],alpha=0.5,label="Z")
        plt.scatter(xs.flatten(),zs.flatten(),alpha=0.5,label="XZ",s=1)
        plt.scatter(ys.flatten(),zs.flatten(),alpha=0.5,label="YZ",s=1)
        plt.legend()
        plt.savefig("cart_v_Z.png")

        del_z = []
        del_xz = []
        del_yz = []

        for i in range(len(zs)):
            gx = np.diff(zs[i]+1e-5)/(np.diff(xs[i])+1e-5)
            gy = np.diff(zs[i]+1e-5)/(np.diff(ys[i])+1e-5)
            del_xz.append(gx)
            del_yz.append(gy)
            gz = np.sqrt(gx**2 + gy**2)
            gz = np.clip(gz,0,100)
            del_z.append(gz)

        plt.close()


        del_z = np.array(del_z)
        del_z = np.round(del_z,6)

        del_xz = np.array(del_xz)
        del_yz = np.array(del_yz)
        del_xz = del_xz.flatten()
        del_yz = del_yz.flatten()

        lim_x = min(len(del_xz),len(xs.flatten()))
        lim_y = min(len(del_yz),len(ys.flatten()))

        del_xz = del_xz[:lim_x]
        xs_t = xs.flatten()[:lim_x]

        del_yz = del_yz[:lim_y]
        ys_t = ys.flatten()[:lim_y]

        # plt.scatter(xs_t,del_xz,label="X",alpha=0.5)
        # plt.scatter(ys_t,del_yz,label="Y",alpha=0.5)
        # plt.legend()
        # plt.savefig("delta.png")
        # plt.close()

        # gaussian filter
        
        del_z = ndimage.gaussian_filter(del_z, sigma=3)
        del_z = abs(del_z)
        
        
        # convert to grayscale
        del_z = (del_z - del_z.min()) / (del_z.max() - del_z.min())
        del_z = 255 * del_z

        del_z_inv = np.flip(del_z,0)
        plt.imshow(del_z_inv,cmap='gray')

        print("Landscape shape: ",landscape.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(landscape)
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

        # remove all z values greater than 1
        landscape = landscape[landscape[:,2] < 1]

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