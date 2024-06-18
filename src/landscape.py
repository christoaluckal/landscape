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
from scipy.spatial import KDTree
import time

def getGradient(point,neighbors):
    gx = np.diff(neighbors[:,2]+1e-5)/(np.diff(neighbors[:,0])+1e-5)
    gy = np.diff(neighbors[:,2]+1e-5)/(np.diff(neighbors[:,1])+1e-5)

    gz = np.sqrt(gx**2 + gy**2)
    gz = np.median(gz)
    gz = np.clip(gz,0,10)

    # print(f"Point: {point} Neighbors: {neighbors} gx: {gx} gy: {gy} gz: {gz}")

    return gz

def cart2Im(
    cartX=None,
    cartY=None,
    extent=None,
    resolution=None,
):
    row = int((extent[1] - cartY) / resolution)
    col = int((cartX - extent[0]) / resolution)

    return row, col

def createEmptyImage(
        extent=5,
        resolution=None
):
    num_rows = int(2*extent / resolution)
    num_cols = int(2*extent / resolution)
    return np.zeros((num_rows,num_cols))


def gradient2D(landscape,count):
    landscape_tree = KDTree(landscape)
    start = time.time()
    new_landscape = []

    prev_landscape = np.copy(landscape)
    pcd_prev = o3d.geometry.PointCloud()
    pcd_prev.points = o3d.utility.Vector3dVector(prev_landscape)
    o3d.visualization.draw_geometries([pcd_prev])

    image = createEmptyImage(extent=4,resolution=0.1)

    for i in range(len(landscape)):
        point = landscape[i]
        r = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        if r < 2:
            new_landscape.append([point[0],point[1],0])
            continue

        closest = landscape_tree.query(point,k=30)
        # closest = getClosestN(landscape,point,n=5)
        closest_idx = closest[1][1:]
        closest = landscape[closest_idx]
        getGradient(point,closest)
        new_landscape.append([point[0],point[1],getGradient(point,closest)])

        row,col = cart2Im(cartX=point[0],cartY=point[1],extent=[4,4],resolution=0.1)
        image[row,col] = getGradient(point,closest)

    new_landscape = np.array(new_landscape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_landscape)
    o3d.visualization.draw_geometries([pcd])

    plt.imshow(image,cmap='gray')
    plt.colorbar()
    plt.show()

    
    end = time.time()
    print("Time: ",end-start)

def filter_distance(landscape,distance=5):
    # convert to polar
    r = np.sqrt(landscape[:,0]**2 + landscape[:,1]**2 + landscape[:,2]**2)
    r_mask = r > 1
    x_mask = np.abs(landscape[:,0]) < distance 
    y_mask = np.abs(landscape[:,1]) < distance
    r_mask = np.logical_and(r_mask,np.logical_and(x_mask,y_mask))
    # r_mask = np.logical_and(x_mask,y_mask)
    
    # r_mask = r < distance
    landscape = landscape[r_mask]

    return landscape

def ouster_cb():
    with open("landscape.pkl","rb") as f:
        landscape = pickle.load(f)

    # remove all z values greater than 1
    landscape = landscape[landscape[:,2] < 1]

    # get median z value
    min_z = np.median(landscape[:,2])
    landscape[:,2] = landscape[:,2] - min_z

    # filter distance
    landscape = filter_distance(landscape,distance=4)

    # ind = np.lexsort((landscape[:,1],landscape[:,0]))

    # landscape = landscape[ind]

    landscape = landscape[::5]

    # n = int(np.sqrt(len(landscape)))

    # landscape = landscape[:n**2]

    
    print("X min: ",landscape[:,0].min())
    print("X max: ",landscape[:,0].max())
    print("Y min: ",landscape[:,1].min())
    print("Y max: ",landscape[:,1].max())

    gradient2D(landscape,1)
        

ouster_cb()