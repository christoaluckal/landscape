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
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    x,y,z = pyproj.transformer.Transformer.from_proj(lla, ecef).transform(lon, lat, alt)

    return x, y, z

# lla_list = []



# for i in tqdm(range(0, dataset.RasterYSize, 50)):
#     for j in range(0, dataset.RasterXSize, 50):
#         lat, lon = pixel2latlon(i, j, gt[0], gt[1], gt[2], gt[3], gt[4], gt[5])
#         lla_list.append((lat, lon, ele_masked[i,j]))

center = [ele_masked.shape[0]//2, ele_masked.shape[1]//2]
lat_0,lon_0 = pixel2latlon(center[0], center[1], gt[0], gt[1], gt[2], gt[3], gt[4], gt[5])
ele_0 = ele_masked[center[0], center[1]]

ele_masked = (ele_masked - ele_masked.min()) / (ele_masked.max() - ele_masked.min())

plt.matshow(ele_masked)
plt.show()

# x_0, y_0, z_0 = gps_to_ecef_pyproj(lat_0, lon_0, ele_0)

# ele_masked = ele_masked - ele_0

# xyz = []

# for i in tqdm(range(0, dataset.RasterYSize, 100)):
#     for j in range(0, dataset.RasterXSize, 100):
#         lat, lon = pixel2latlon(j,i, gt[0], gt[1], gt[2], gt[3], gt[4], gt[5])
#         ele = ele_masked[i,j]
#         x, y, z = gps_to_ecef_pyproj(lat, lon, ele)
#         x = x - x_0
#         y = y - y_0
#         z = z - z_0
#         xyz.append((x,y,z))

# xyz = np.array(xyz)
# np.save("xyz.npy", xyz)
