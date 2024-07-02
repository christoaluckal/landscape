from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pyproj.transformer
from tqdm import tqdm


def pixel2latlon(x, y,gt_0, gt_1, gt_2, gt_3, gt_4, gt_5):

    """Returns lat lon from pixel"""
    lon = gt_0 + x * gt_1 + y * gt_2
    lat = gt_3 + x * gt_4 + y * gt_5
    return lat, lon

def ecef2enu(reference_ECEF=None,reference_LatLongAlt=None,candidate_ECEF=None):
    G_phi = reference_LatLongAlt[0]
    G_lambda = reference_LatLongAlt[1]
    G_h = reference_LatLongAlt[2]
    

    if type(candidate_ECEF) == np.ndarray and type(reference_ECEF) == np.ndarray:

        pass

    else:
        reference_ECEF = np.array(reference_ECEF)
        candidate_ECEF = np.array(candidate_ECEF)

    R = np.array([
        [-np.sin(G_lambda), np.cos(G_lambda), 0],
        [-np.sin(G_phi)*np.cos(G_lambda), -np.sin(G_phi)*np.sin(G_lambda), np.cos(G_phi)],
        [np.cos(G_phi)*np.cos(G_lambda), np.cos(G_phi)*np.sin(G_lambda), np.sin(G_phi)]
        ])
    P = candidate_ECEF - reference_ECEF
    ENU = np.dot(R,P.T).T
    return ENU
        
    
    # gX = reference_ECEF[0]
    # gY = reference_ECEF[1]
    # gZ = reference_ECEF[2]

    # cX = candidate_ECEF[0]
    # cY = candidate_ECEF[1]
    # cZ = candidate_ECEF[2]

    # E = -1*(cX-gX)*np.sin(G_lambda) + (cY-gY)*np.cos(G_lambda)
    # N = -1*(cX-gX)*np.cos(G_lambda)*np.sin(G_phi) - (cY-gY)*np.sin(G_lambda)*np.sin(G_phi) + (cZ-gZ)*np.cos(G_phi)
    # U = (cX-gX)*np.cos(G_lambda)*np.cos(G_phi) + (cY-gY)*np.sin(G_lambda)*np.cos(G_phi) + (cZ-gZ)*np.sin(G_phi)

    # ENU = np.vstack((E,N,U))
    # return ENU.T


    
def latlon2pixel(lat, lon):
    """Returns pixel coordinates from lat lon"""
    x = (lon - gt[0]) / gt[1]
    y = (lat - gt[3]) / gt[5]
    return x, y
    # pass

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

def gps_to_ecef_pyproj(lat, lon, alt):
    lat = np.radians(lat)
    lon = np.radians(lon)
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=True)
    x,y,z = pyproj.transformer.Transformer.from_proj(lla, ecef).transform(lon, lat, alt, radians=True)

    return z,x,y

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

ex1_x, ex1_y, ex1_z = gps_to_ecef_pyproj(ex1_lat, ex1_lon, ele[im_extent[0][0], im_extent[0][1]])
ex3_x, ex3_y, ex3_z = gps_to_ecef_pyproj(ex3_lat, ex3_lon, ele[im_extent[2][0], im_extent[2][1]])
ex7_x, ex7_y, ex7_z = gps_to_ecef_pyproj(ex7_lat, ex7_lon, ele[im_extent[6][0], im_extent[6][1]])
ex9_x, ex9_y, ex9_z = gps_to_ecef_pyproj(ex9_lat, ex9_lon, ele[im_extent[8][0], im_extent[8][1]])
cent_x, cent_y, cent_z = gps_to_ecef_pyproj(center_lat, center_lon, ele[im_extent[4][0], im_extent[4][1]])

print(f"Ex1 LatLon: {ex1_lat}, {ex1_lon}")
print(f"Ex3 LatLon: {ex3_lat}, {ex3_lon}")
print(f"Ex7 LatLon: {ex7_lat}, {ex7_lon}")
print(f"Ex9 LatLon: {ex9_lat}, {ex9_lon}")

print(f"Ex1-3, {ex3_x - ex1_x}, {ex3_y - ex1_y}")
print(f"Ex1-7, {ex7_x - ex1_x}, {ex7_y - ex1_y}")

lats = []
lons = []
alts = []

for i in tqdm(range(0,ele_masked.shape[0],5)):
    for j in range(0,ele_masked.shape[1],5):
        lat, lon = pixel2latlon(j,i, *gt)
        alt = ele_masked[i, j]
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)


lats = np.array(lats)
lons = np.array(lons)
alts = np.array(alts)

x,y,z = gps_to_ecef_pyproj(lats, lons, alts)


xyz = np.vstack((x,y,z)).T

ENUs = ecef2enu([cent_x, cent_y, cent_z],[center_lat, center_lon, 0], xyz)


# ENUs[:,[0,1,2]] = ENUs[:,[0,1,2]]
# ENUs[:,[0,1,2]] = ENUs[:,[0,2,1]]
# ENUs[:,[0,1,2]] = ENUs[:,[1,0,2]]
# ENUs[:,[0,1,2]] = ENUs[:,[1,2,0]]
# ENUs[:,[0,1,2]] = ENUs[:,[2,0,1]]
ENUs[:,[0,1,2]] = ENUs[:,[2,1,0]]



printBounds(ENUs)

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ENUs)
o3d.visualization.draw_geometries([pcd])



