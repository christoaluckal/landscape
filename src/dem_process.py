from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pyproj.transformer
from tqdm import tqdm


def pixel2latlon(x, y,gt_0, gt_1, gt_2, gt_3, gt_4, gt_5):

    # swap x and y
    # x, y = y, x

    """Returns lat lon from pixel"""
    lon = gt_0 + x * gt_1 + y * gt_2
    lat = gt_3 + x * gt_4 + y * gt_5
    return lat, lon

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

plt.imshow(ele_masked)
plt.show()

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

ex1_x, ex1_y, ex1_z = gps_to_ecef_pyproj(ex1_lat, ex1_lon, ele[im_extent[0][0], im_extent[0][1]])
ex3_x, ex3_y, ex3_z = gps_to_ecef_pyproj(ex3_lat, ex3_lon, ele[im_extent[2][0], im_extent[2][1]])
ex7_x, ex7_y, ex7_z = gps_to_ecef_pyproj(ex7_lat, ex7_lon, ele[im_extent[6][0], im_extent[6][1]])
ex9_x, ex9_y, ex9_z = gps_to_ecef_pyproj(ex9_lat, ex9_lon, ele[im_extent[8][0], im_extent[8][1]])

print(f"Ex1 LatLon: {ex1_lat}, {ex1_lon}")
print(f"Ex3 LatLon: {ex3_lat}, {ex3_lon}")
print(f"Ex7 LatLon: {ex7_lat}, {ex7_lon}")
print(f"Ex9 LatLon: {ex9_lat}, {ex9_lon}")

print(f"Ex1-3, {ex3_x - ex1_x}, {ex3_y - ex1_y}")
print(f"Ex1-7, {ex7_x - ex1_x}, {ex7_y - ex1_y}")

lats = []
lons = []
alts = []

for i in range(ele_masked.shape[0]):
    for j in range(ele_masked.shape[1]):
        lat, lon = pixel2latlon(i, j, *gt)
        alt = ele_masked[i, j]
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)


lats = np.array(lats)
lons = np.array(lons)
alts = np.array(alts)



