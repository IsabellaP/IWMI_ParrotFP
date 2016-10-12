import os
import gdal
import numpy as np
import pandas as pd
import fnmatch
from datetime import datetime
from netCDF4 import Dataset


def read_nc(path, datestr, lon, lat, key='SWI_040', 
            start_date=datetime(2010,1,1), end_date=datetime(2015, 12, 31)):
    """
    Parameters:
    -----------
    path : str
        Path to nc-data
    datestr : dict
        position of date in filename, e.g. {'year': [0,4], 'month': [4,6], 
        'day': [6,8]}
    lon, lat : float, optional
        Longitude and latitude of point of interest
    key : str, optional
        Parameter to be read (name as in nc-files). Default: SWI_040
    start_date, end_date : datetime.datetime, optional
        Start and end timestamp of time series
        
    Returns:
    --------
    data : dict
        Dataset
    """

    folders = os.listdir(path)
    timestamp_array = []

    for fname in sorted(folders):
        year = int(fname[datestr['year'][0]:datestr['year'][1]])
        month = int(fname[datestr['month'][0]:datestr['month'][1]])
        day = int(fname[datestr['day'][0]:datestr['day'][1]])
        timestamp_array.append(datetime(year, month, day))

    timestamp_array = np.array(timestamp_array)
    date_idx = np.where((timestamp_array>=start_date) &
                        (timestamp_array<=end_date))[0]

    folderlist = np.array(folders)[date_idx]

    param_data = []
    for folder in sorted(folderlist):
        fpath = os.path.join(path, folder)
        fname = fnmatch.filter(os.listdir(fpath), '*.nc')[0]
        with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
            lons = ncfile.variables['lon'][:]
            lats = ncfile.variables['lat'][:]

            # find nearest lonlat
            nearest_lon = find_nearest(lons, lon)
            nearest_lat = find_nearest(lats, lat)
            lon_idx = np.where(lons == nearest_lon)[0]
            lat_idx = np.where(lats == nearest_lat)[0]
            param_data.append(ncfile.variables[key][lat_idx, lon_idx][0][0])

    param_data = np.array(param_data)
    df = pd.DataFrame(param_data, index=timestamp_array[date_idx],
                      columns=[key])

    return df


def read_tiff(path, datestr, lon, lat, start_date=datetime(2010,1,1), 
              end_date=datetime(2015, 12, 31)):
    
    ts = []
    for tiff in os.listdir(path):
        year = int(tiff[datestr['year'][0]:datestr['year'][1]])
        month = int(tiff[datestr['month'][0]:datestr['month'][1]])
        day = int(tiff[datestr['day'][0]:datestr['day'][1]])
        
        if (datetime(year, month, day) < start_date or 
            datetime(year, month, day) > end_date):
            continue
        
        arr, lons, lats = open_tiff(path, tiff)
        nearest_lon = find_nearest(lons, lon)
        nearest_lat = find_nearest(lats, lat)
        lon_idx = np.where(lons == nearest_lon)[0]
        lat_idx = np.where(lats == nearest_lat)[0]
        ts.append(arr[lat_idx, lon_idx])
        
    ts = np.array(ts)
    return ts        


def open_tiff(srcpath, fname):
    
    src_filename = os.path.join(srcpath, fname)

    # Opens source dataset
    src_ds = gdal.Open(src_filename)
    band = src_ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    latsize, lonsize = arr.shape
    
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    gt = src_ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5] 
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]
    
    px = (maxx-minx)/lonsize
    py = (maxy-miny)/latsize
    
    lons = np.arange(minx, maxx, px)
    lats = np.arange(miny, maxy, py)
        
    return arr, lons, lats


def find_nearest(array, element):
    idx = (np.abs(array - element)).argmin()
    return array[idx]


if __name__ == '__main__':
    
    ndvi_path = "E:\\poets\\RAWDATA\\NDVI_8daily_500\\"
    datestr = {'year': [5,9], 'month': [9,11], 'day': [11,13]}
    lon = 75.75
    lat = 20.15
    ndvi_ts = read_tiff(ndvi_path, datestr, lon, lat)
    
    swi_path = "E:\\poets\\RAWDATA\\SWI_daily_01\\"
    datestr = {'year': [14,18], 'month': [18,20], 'day': [20,22]}
    key = ['SWI_001', 'SWI_040']
    swi_ts = read_nc(swi_path, datestr, lon, lat, key)
    
    