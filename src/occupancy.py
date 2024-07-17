import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pickle
from scipy.spatial import KDTree
import cv2
import tqdm
import argparse

map_templet = '''
image: IMAGE_PATH
resolution: IMAGE_RES
origin: [IMAGE_OX, IMAGE_OY, 0.0]
occupied_thresh: 0.65
free_thresh: 0.196
negate: 0
'''


parser = argparse.ArgumentParser()
parser.add_argument("--nav2", action="store_true", help="Use nav2")
parser.add_argument("--max_gradient", type=int, help="Max gradient", default=10)
parser.add_argument("--distance", type=int, help="Filter distance", default=50)
parser.add_argument("--skips", type=int, help="Skip every n points", default=10)
parser.add_argument("--resolution", type=float, help="Resolution (m/px) of the image", default=0.1)
parser.add_argument("--elevation_filter", type=int, help="Elevation filter in Z", default=5)
parser.add_argument("--polar_filter", type=int, help="Polar distance filter", default=2)
parser.add_argument("--kd_neighbors", type=int, help="Number of neighbors for KDTree", default=30)
parser.add_argument("--window_size", type=int, help="Window size for max filter", default=3)
parser.add_argument("--sigma", type=int, help="Sigma for gaussian filter", default=3)
parser.add_argument("--dilate_iterations", type=int, help="Number of dilate iterations", default=3)
parser.add_argument("--dilation_size", type=int, help="Dilation size", default=3)
parser.add_argument("--pkl_path", type=str, help="Path to pkl file", default="workspace.pkl")

args = parser.parse_args()


def getGradient(neighbors):
    gx = np.diff(neighbors[:,2]+1e-5)/(np.diff(neighbors[:,0])+1e-5)
    gy = np.diff(neighbors[:,2]+1e-5)/(np.diff(neighbors[:,1])+1e-5)

    gz = np.sqrt(gx**2 + gy**2)
    gz = np.median(gz)
    gz = np.clip(gz,0,args.max_gradient)

    theta_x = np.median(np.arctan(gx))
    theta_y = np.median(np.arctan(gy))

    angle = abs(max(theta_x,theta_y))
    angle = min(angle,np.radians(30))

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

    image = createEmptyImage(extent=dist,resolution=args.resolution)
    angle_image = createEmptyImage(extent=dist,resolution=args.resolution)
    elevation_image = createEmptyImage(extent=dist,resolution=args.resolution)

    for i in tqdm.tqdm(range(len(landscape))):
        point = landscape[i]
        r = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        if r < args.polar_filter:
            continue

        closest = landscape_tree.query(point,k=args.kd_neighbors)
        closest_idx = closest[1][1:]
        closest = landscape[closest_idx]

        row,col = cart2Im(cartX=point[0],cartY=point[1],extent=[dist,dist],resolution=args.resolution)
        e,angle = getGradient(closest)
        image[row,col] = e
        angle_image[row,col] = angle
        elevation_image[row,col] = point[2]

 

    if args.nav2:
        image = cv2.dilate(image, np.ones((args.dilation_size,args.dilation_size),np.uint8), iterations=args.dilate_iterations)
        image = ndimage.gaussian_filter(image, sigma=args.sigma)

        image = (image - image.min())/(image.max()-image.min())
        image = 255*(1-image)

        mask = image > 240
        image[mask] = 255
        image[~mask] = 0

        # image = cv2.dilate(image, np.ones((11,11),np.uint8), iterations=2)
        image = ndimage.gaussian_filter(image, sigma=args.sigma)
        
        image = np.uint8(image)

        plt.imshow(image)
        plt.show()


    else:
        image = cv2.dilate(image, np.ones((args.dilation_size,args.dilation_size),np.uint8), iterations=args.dilate_iterations)
        
        elevation_image = windowMax(elevation_image,window_size=args.window_size)


        # image = np.uint8(image)
        plt.matshow(image)
        plt.colorbar()
        plt.show()

        angle_image = windowMax(angle_image,window_size=3)

        angle_image = np.degrees(angle_image)
        plt.matshow(angle_image)
        plt.colorbar()
        plt.show()

    cv2.imwrite("landscape.pgm",image)

    yaml = map_templet.replace("IMAGE_PATH","landscape.pgm")
    yaml = yaml.replace("IMAGE_RES",f"{args.resolution}")
    yaml = yaml.replace("IMAGE_OX",f"{-dist}")
    yaml = yaml.replace("IMAGE_OY",f"{-dist}")

    with open("landscape.yaml","w") as f:
        f.write(yaml)

    cv2.imwrite("landscape.png",image)


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

def generateOccupancyFiles(filter_dist=50,skips=10):
    # with open("workspace.pkl","rb") as f:
    #     landscape = pickle.load(f)

    with open(args.pkl_path,"rb") as f:
        landscape = pickle.load(f)

    # remove all z values greater than elevation filter
    landscape = landscape[landscape[:,2] < args.elevation_filter]

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
        

if __name__ == "__main__":
    generateOccupancyFiles(filter_dist=args.distance,skips=args.skips)
