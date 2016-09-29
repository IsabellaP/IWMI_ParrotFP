import os
import numpy as np
import pandas as pd
import fnmatch
from datetime import datetime
from netCDF4 import Dataset


def read_ts(path, datestr, lon, lat, key='SWI_040', start_date=datetime(2010,1,1), 
            end_date=datetime(2015, 12, 31)):
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


def find_nearest(array, element):
    idx = (np.abs(array - element)).argmin()
    return array[idx]

