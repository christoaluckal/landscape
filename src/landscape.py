import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
# from scipy.ndimage import gaussian_filter1d
from numpy import convolve
import pickle
from scipy.spatial import KDTree
import time
import cv2
import sys
import tqdm

map_templet = '''
image: IMAGE_PATH
resolution: IMAGE_RES
origin: [IMAGE_OX, IMAGE_OY, 0.0]
occupied_thresh: 0.65
free_thresh: 0.196
negate: 0
'''

nav2_flag = bool(int(sys.argv[1]))

def getGradient(point,neighbors):
    gx = np.diff(neighbors[:,2]+1e-5)/(np.diff(neighbors[:,0])+1e-5)
    gy = np.diff(neighbors[:,2]+1e-5)/(np.diff(neighbors[:,1])+1e-5)

    gz = np.sqrt(gx**2 + gy**2)
    gz = np.median(gz)
    gz = np.clip(gz,0,10)

    theta_x = np.median(np.arctan(gx))
    theta_y = np.median(np.arctan(gy))

    angle = abs(max(theta_x,theta_y))
    angle = min(angle,np.radians(30))

    # print(f"Point: {point} Neighbors: {neighbors} gx: {gx} gy: {gy} gz: {gz}")

    return gz, angle

def windowMax(image,window_size=3):
    im_copy = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i < window_size or j < window_size or i > image.shape[0]-window_size or j > image.shape[1]-window_size:
                im_copy[i,j] = image[i,j]
                continue
            im_copy[i,j] = np.max(image[i-window_size:i+window_size,j-window_size:j+window_size])

    return im_copy
   

def img2Cart(
    row=None,
    col=None,
    extent=None,
    res=None
):
    x = col*res-extent[0]
    y = extent[1]-row*res

    return x,y

def cart2Im(
    cartX=None,
    cartY=None,
    extent=None,
    resolution=None,
):
    # row = int((extent[1] - cartY) / resolution)
    # col = int((cartX - extent[0]) / resolution)

    row = int((extent[1]-cartY)/resolution)
    col = int((extent[1]+cartX)/resolution)

    return row, col

def createEmptyImage(
        extent=5,
        resolution=None
):
    num_rows = int(2*extent / resolution)
    num_cols = int(2*extent / resolution)
    return np.zeros((num_rows,num_cols))


def gradient2D(landscape,dist):

    # nav2 = False
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(landscape)
    o3d.visualization.draw_geometries([pcd])

    landscape_tree = KDTree(landscape)
    start = time.time()

    image = createEmptyImage(extent=dist,resolution=0.1)
    angle_image = createEmptyImage(extent=dist,resolution=0.1)

    for i in tqdm.tqdm(range(len(landscape))):
        point = landscape[i]
        r = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        if r < 2:
            continue

        closest = landscape_tree.query(point,k=100)
        # closest = getClosestN(landscape,point,n=5)
        closest_idx = closest[1][1:]
        closest = landscape[closest_idx]

        row,col = cart2Im(cartX=point[0],cartY=point[1],extent=[dist,dist],resolution=0.1)
        e,angle = getGradient(point,closest)
        image[row,col] = e
        angle_image[row,col] = angle

    # binarize image
    # mask = image < 1
    # image[mask] = 0
    

    if nav2_flag:
        image = cv2.dilate(image, np.ones((3,3),np.uint8), iterations=1)
        image = ndimage.gaussian_filter(image, sigma=1)

        image = (image - image.min())/(image.max()-image.min())


        image = 255*(1-image)
        image = np.rot90(image,3)


        mask = image > 245
        image[mask] = 255
        image[~mask] = 0
        
        image = np.uint8(image)

        plt.imshow(image)
        plt.show()


    else:
        image = cv2.dilate(image, np.ones((3,3),np.uint8), iterations=1)

        image = (image - image.min())/(image.max()-image.min())
        image = 255*(image)
        image = ndimage.gaussian_filter(image, sigma=3)

        image = np.uint8(image)
        plt.imshow(image)
        plt.show()

        angle_image = windowMax(angle_image,window_size=3)
        # angle_image = (angle_image - angle_image.min())/(angle_image.max()-angle_image.min())
        # angle_image = 255*angle_image

        # angle_image = np.uint8(angle_image)
        angle_image = np.degrees(angle_image)
        plt.matshow(angle_image)
        plt.colorbar()
        plt.show()

        # print(cart2Im(0,0,[dist,dist],0.1))
        # print(cart2Im(5,5,[dist,dist],0.1))
        # print(cart2Im(5,-5,[dist,dist],0.1))
        # print(cart2Im(-5,-5,[dist,dist],0.1))
        # print(cart2Im(-5,5,[dist,dist],0.1))

        # print(img2Cart(50,100,[dist,dist],0.1))
        # print(img2Cart(100,150,[dist,dist],0.1))
        # print(img2Cart(150,100,[dist,dist],0.1))
        # print(img2Cart(100,50,[dist,dist],0.1))
    
 

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

def ouster_cb(filter_dist=5,skips=5):
    with open("workspace.pkl","rb") as f:
        landscape = pickle.load(f)

    # remove all z values greater than 1
    landscape = landscape[landscape[:,2] < 50]

    # get median z value
    min_z = np.median(landscape[:,2])
    landscape[:,2] = landscape[:,2] - min_z

    # filter distance
    landscape = filter_distance(landscape,distance=filter_dist)

    # ind = np.lexsort((landscape[:,1],landscape[:,0]))

    # landscape = landscape[ind]
    print("Landscape shape: ",landscape.shape)
    landscape = landscape[::skips]
    print("Landscape shape: ",landscape.shape)


    print(f"Time:{gradient2D(landscape,filter_dist)}")
    print("*"*50,"\n")
        
skips = [10]
for skip in skips:
    ouster_cb(filter_dist=50,skips=skip)
# ouster_cb(filter_dist=5)