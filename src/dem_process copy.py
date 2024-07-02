from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pyproj.transformer
from tqdm import tqdm
from scipy.spatial import KDTree


def pixel2latlon(x, y,gt_0, gt_1, gt_2, gt_3, gt_4, gt_5):

    """Returns lat lon from pixel"""
    lon = gt_0 + x * gt_1 + y * gt_2
    lat = gt_3 + x * gt_4 + y * gt_5
    return lat, lon


def printBounds(array):
    x_min = np.min(array[:,0])
    x_max = np.max(array[:,0])
    y_min = np.min(array[:,1])
    y_max = np.max(array[:,1])
    z_min = np.min(array[:,2])
    z_max = np.max(array[:,2])

    print(f"X: {x_min} - {x_max}")
    print(f"Y: {y_min} - {y_max}")
    print(f"Z: {z_min} - {z_max}")

""" Convert GPS (Latitude and Longitude) to XYZ wrt a set Lat Long as origin """


def geodedic_to_ecef( lati, longi, alti ):
    """ lati in degrees, longi in degrees. alti in meters (mean sea level) """
    # Adopted from https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    phi = lati / 180. * np.pi
    lambada = longi / 180. * np.pi
    h = alti

    #N = 6371000 #in meters
    e = 0.081819191 #earth ecentricity
    q = np.sin( phi )
    N = 6378137.0 / np.sqrt( 1 - e*e * q*q )
    X = (N + h) * np.cos( phi ) * np.cos( lambada )
    Y = (N + h) * np.cos( phi ) * np.sin( lambada )
    Z = (N*(1-e*e) + h) * np.sin( phi )

    return X,Y,Z

def compute_ecef_to_enu_transform( lati_r, longi_r ):
    """ Computes a matrix_3x3 which transforms a ecef diff-point to ENU (East-North-Up)
        Needs as input the latitude and longitude (in degrees) of the reference point
    """
    # Adopted from https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

    phi = lati_r / 180. * np.pi
    lambada = longi_r / 180. * np.pi

    cp = np.cos( phi ) #cos(phi)
    sp = np.sin( phi ) #cos(phi)
    cl = np.cos( lambada )
    sl = np.sin( lambada )

    T = np.zeros( (3,3), dtype='float64' )
    T[0,0] = -sl
    T[0,1] = cl
    T[0,2] = 0

    T[1,0] = -sp * cl
    T[1,1] = -sp * sl
    T[1,2] = cp

    T[2,0] = cp * cl
    T[2,1] = cp * sl
    T[2,2] = sp

    T_enu_ecef = T
    return T_enu_ecef



def rotate(points,rpy=[0,0,0]):
    """ Rotate points by roll, pitch, yaw """
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]

    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    R = np.array([
        [np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), -np.sin(pitch)],
        [np.sin(roll)*np.sin(pitch)*np.cos(yaw) - np.cos(roll)*np.sin(yaw), np.sin(roll)*np.sin(pitch)*np.sin(yaw) + np.cos(roll)*np.cos(yaw), np.sin(roll)*np.cos(pitch)],
        [np.cos(roll)*np.sin(pitch)*np.cos(yaw) + np.sin(roll)*np.sin(yaw), np.cos(roll)*np.sin(pitch)*np.sin(yaw) - np.sin(roll)*np.cos(yaw), np.cos(roll)*np.cos(pitch)]
    ])

    return np.dot(R,points.T).T

def computeOrientation(tree: KDTree, k: int = 50):

    avg_roll = 0
    avg_pitch = 0
    avg_yaw = 0
    trials = 100

    for _ in range(trials):
        idx = np.random.randint(0,tree.data.shape[0])
        point = tree.data[idx]

        closest_k = tree.query(point, k=k)

        reference = point
        closest_k = tree.data[closest_k[1]]


        
        for i in range(k):
            delta = closest_k[i] - reference
            roll = np.arctan2(delta[1],delta[2])
            pitch = np.arctan2(delta[0],delta[2])
            yaw = np.arctan2(delta[0],delta[1])

            avg_roll += roll
            avg_pitch += pitch
            avg_yaw += yaw

    
    avg_roll /= k*trials
    avg_pitch /= k*trials
    avg_yaw /= k*trials

    avg_roll = np.degrees(avg_roll)
    avg_pitch = np.degrees(avg_pitch)
    avg_yaw = np.degrees(avg_yaw)

    print(f"Roll: {avg_roll}, Pitch: {avg_pitch}, Yaw: {avg_yaw}")





dataset = gdal.Open("space.tif", gdal.GA_ReadOnly)

print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                            dataset.GetDriver().LongName))
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))
print("Projection is {}".format(dataset.GetProjection()))
geotransform = dataset.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))


band = dataset.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min,max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min,max))

if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))


ele = band.ReadAsArray()
mask = ele < min
ele_masked = np.copy(ele)
ele_masked[mask] = min
ele = np.copy(ele_masked)
# ele_norm = (ele_masked - min) / (max - min)

# plt.imshow(ele_masked)
# plt.show()

gt = dataset.GetGeoTransform()


mid_row = ele.shape[0] // 2
mid_col = ele.shape[1] // 2

im_extent = [
    [0,0],
    [0, mid_col],
    [0, 2*mid_col - 1],
    [mid_row, 0],
    [mid_row, mid_col],
    [mid_row, 2*mid_col - 1],
    [2*mid_row - 1, 0],
    [2*mid_row - 1, mid_col],
    [2*mid_row - 1, 2*mid_col - 1]
]

latlon_extent = []
for i in im_extent:
    latlon_extent.append(i)


ex1_lat, ex1_lon = pixel2latlon(latlon_extent[0][0], latlon_extent[0][1], *gt)
ex3_lat, ex3_lon = pixel2latlon(latlon_extent[2][0], latlon_extent[2][1], *gt)
ex7_lat, ex7_lon = pixel2latlon(latlon_extent[6][0], latlon_extent[6][1], *gt)
ex9_lat, ex9_lon = pixel2latlon(latlon_extent[8][0], latlon_extent[8][1], *gt)
center_lat, center_lon = pixel2latlon(latlon_extent[4][0], latlon_extent[4][1], *gt)
print(f"Center LatLon: {center_lat}, {center_lon}, {ele[im_extent[4][0], im_extent[4][1]]}")

res_x = gt[1]
res_y = gt[5]

X = []
Y = []
Z = []

Xr,Yr,Zr = geodedic_to_ecef(center_lat,center_lon,ele[im_extent[4][0], im_extent[4][1]])
T = compute_ecef_to_enu_transform(ex7_lat,ex7_lon)

for i in tqdm(range(0,ele_masked.shape[0],5)):
    for j in range(0,ele_masked.shape[1],5):
        lat,lon = pixel2latlon(j,i,*gt)
        alt = ele_masked[i,j]
        
        Xp,Yp,Zp = geodedic_to_ecef(lat,lon,alt)
        X.append(Xp)
        Y.append(Yp)
        Z.append(Zp)


X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

delta = np.array([X-Xr,Y-Yr,Z-Zr])
p = np.dot(T,delta).T

point_tree = KDTree(p)

computeOrientation(point_tree)
# p = rotate(p,[90,0,0])

# reflect about Y axis
# p[:,2] = -p[:,2]

import pickle

with open('workspace.pkl', 'wb') as f:
    pickle.dump(p, f)

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(p)

o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("space_2.ply", pcd)