from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pyproj.transformer
from tqdm import tqdm


def pixel2latlon(x, y,gt_0, gt_1, gt_2, gt_3, gt_4, gt_5):
    """Returns lat lon from pixel"""
    lat = gt_0 + x * gt_1 + y * gt_2
    lon = gt_3 + x * gt_4 + y * gt_5
    return lat, lon

def latlon2pixel(lat, lon):
    """Returns pixel coordinates from lat lon"""
    x = (lon - gt[0]) / gt[1]
    y = (lat - gt[3]) / gt[5]
    return x, y
    # pass



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
# ele_norm = (ele_masked - min) / (max - min)

plt.matshow(ele_masked)
plt.show()

gt = dataset.GetGeoTransform()

def gps_to_ecef_pyproj(lat, lon, alt):
    lat = np.radians(lat)
    lon = np.radians(lon)
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    x,y,z = pyproj.transformer.Transformer.from_proj(lla, ecef).transform(lon, lat, alt)

    return x, y, z

top_left_lat, top_left_lon = pixel2latlon(0, 0, gt[0], gt[1], gt[2], gt[3], gt[4], gt[5])

top_left_x, top_left_y, top_left_z = gps_to_ecef_pyproj(top_left_lat, top_left_lon, ele[0,0])

xyz = []

image = []
lats = []
lons = []
alts = []

for i in tqdm(range(0, ele.shape[0], 10)):
    row = []
    for j in range(0,ele.shape[1],10):
        row.append(ele_masked[i,j])
        lat, lon = pixel2latlon(i, j, gt[0], gt[1], gt[2], gt[3], gt[4], gt[5])
        lats.append(lat)
        lons.append(lon)
        alts.append(ele_masked[i,j])
    image.append(row)

xs,ys,zs = gps_to_ecef_pyproj(lats,lons,alts)
xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)

xs = xs - top_left_x
ys = ys - top_left_y
# zs = zs - top_left_z

for i in range(len(xs)):
    xyz.append([xs[i],ys[i],zs[i]])

xyz = np.array(xyz)

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.visualization.draw_geometries([pcd])
