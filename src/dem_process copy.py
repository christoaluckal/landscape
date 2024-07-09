from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pyproj.transformer
from tqdm import tqdm
from scipy.spatial import KDTree
import math


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

def xyz2llh(x,y,z):
    '''
    https://gis.stackexchange.com/a/292635
    Function to convert xyz ECEF to llh
    convert cartesian coordinate into geographic coordinate
    ellipsoid definition: WGS84
      a= 6,378,137m
      f= 1/298.257

    Input
      x: coordinate X meters
      y: coordinate y meters
      z: coordinate z meters
    Output
      lat: latitude rad
      lon: longitude rad
      h: height meters
    '''
    # --- WGS84 constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    # --- derived constants
    b = a - f*a
    e = math.sqrt(math.pow(a,2.0)-math.pow(b,2.0))/a
    clambda = math.atan2(y,x)
    p = math.sqrt(pow(x,2.0)+pow(y,2))
    h_old = 0.0
    # first guess with h=0 meters
    theta = math.atan2(z,p*(1.0-math.pow(e,2.0)))
    cs = math.cos(theta)
    sn = math.sin(theta)
    N = math.pow(a,2.0)/math.sqrt(math.pow(a*cs,2.0)+math.pow(b*sn,2.0))
    h = p/cs - N
    while abs(h-h_old) > 1.0e-6:
        h_old = h
        theta = math.atan2(z,p*(1.0-math.pow(e,2.0)*N/(N+h)))
        cs = math.cos(theta)
        sn = math.sin(theta)
        N = math.pow(a,2.0)/math.sqrt(math.pow(a*cs,2.0)+math.pow(b*sn,2.0))
        h = p/cs - N
    # llh = {'lon':clambda, 'lat':theta, 'height': h}
    llh = {'lon':np.degrees(clambda), 'lat':np.degrees(theta), 'height': h}
    return llh

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

def ecef2enu( points, lati_r, longi_r, ref_point ):
    T = compute_ecef_to_enu_transform( lati_r, longi_r )
    T = np.linalg.inv(T)
    p = np.dot(T, points.T).T
    if type(ref_point) != np.ndarray:
        ref_point = np.array(ref_point)

    p = p + ref_point

    return p



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

plt.matshow(ele_masked)
plt.show()

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

NW_lat, NW_lon = pixel2latlon(latlon_extent[0][0], latlon_extent[0][1], *gt)
C_lat, C_lon = pixel2latlon(latlon_extent[4][0], latlon_extent[4][1], *gt)
SE_lat, SE_lon = pixel2latlon(latlon_extent[8][0], latlon_extent[8][1], *gt)
C_interp_lat, C_interp_lon = (NW_lat + SE_lat) / 2, (NW_lon + SE_lon) / 2

print(f"NW: {NW_lat}, {NW_lon}, {ele[im_extent[0][0], im_extent[0][1]]}")
print(f"C: {C_lat}, {C_lon}, {ele[im_extent[4][0], im_extent[4][1]]}")
print(f"C_i: {C_interp_lat}, {C_interp_lon}")
print(f"C Error: {C_lat-C_interp_lat}, {C_lon-C_interp_lon}")
print(f"SE: {SE_lat}, {SE_lon}, {ele[im_extent[8][0], im_extent[8][1]]}")

print(f"{NW_lat-SE_lat}, {NW_lon-SE_lon}")

NW_X, NW_Y, NW_Z = geodedic_to_ecef(NW_lat, NW_lon, ele[im_extent[0][0], im_extent[0][1]])
C_X, C_Y, C_Z = geodedic_to_ecef(C_lat, C_lon, ele[im_extent[4][0], im_extent[4][1]])
SE_X, SE_Y, SE_Z = geodedic_to_ecef(SE_lat, SE_lon, ele[im_extent[8][0], im_extent[8][1]])

T = compute_ecef_to_enu_transform(C_lat, C_lon)

X = [NW_X, C_X, SE_X]
Y = [NW_Y, C_Y, SE_Y]
Z = [NW_Z, C_Z, SE_Z]

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

delta = np.array([X-C_X,Y-C_Y,Z-C_Z])
p = np.dot(T,delta).T

NW_x_local, NW_y_local, NW_z_local = p[0]
C_x_local, C_y_local, C_z_local = p[1]
SE_x_local, SE_y_local, SE_z_local = p[2]

# ENU to ECEF
NW_ecef = ecef2enu(np.array([NW_x_local, NW_y_local, NW_z_local]), C_lat, C_lon, [C_X, C_Y, C_Z])
C_ecef = ecef2enu(np.array([C_x_local, C_y_local, C_z_local]), C_lat, C_lon, [C_X, C_Y, C_Z])
SE_ecef = ecef2enu(np.array([SE_x_local, SE_y_local, SE_z_local]), C_lat, C_lon, [C_X, C_Y, C_Z])

print(f"NW ECEF: {NW_ecef}")
print(f"C ECEF: {C_ecef}")
print(f"SE ECEF: {SE_ecef}")

# ECEF to LLH
NW_llh = xyz2llh(*NW_ecef)
C_llh = xyz2llh(*C_ecef)
SE_llh = xyz2llh(*SE_ecef)

print(f"NW LLH: {NW_llh}")
print(f"C LLH: {C_llh}")
print(f"SE LLH: {SE_llh}")

test_X, test_Y = 10, 10
test_Z = C_z_local

test_ecef = ecef2enu(np.array([test_X, test_Y, test_Z]), C_lat, C_lon, [C_X, C_Y, C_Z])
test_llh = xyz2llh(*test_ecef)

print(f"Test LLH: {test_llh}")


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
o3d.io.write_point_cloud("space_2.pcd", pcd)