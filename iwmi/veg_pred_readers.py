import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from datetime import datetime


def read_ts(path, params=['SWI_020'], lon=80.5, lat=6.81,
            start_date=datetime(2010, 1, 1), end_date=datetime(2010, 3, 31)):
    """
    Parameters:
    -----------
    path : str
        Path to nc-data
    params : list of str
        Parameter to be read (name as in nc-files). Default: SWI_020
    lon, lat : float
        Longitude and latitude of point of interest, default: point in Sri
        Lanka.
    start_date, end_date : datetime.datetime
        Start and end timestamp of time series

    Returns:
    --------
    data : pd.DataFrame
        Dataset
    """

    if isinstance(params, basestring):
        params = [params]

    with Dataset(path, "r") as ncfile:
        unit_temps = ncfile.variables['time'].units
        nctime = ncfile.variables['time'][:]
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError:  # Attribute doesn't exist
            cal_temps = u"gregorian"  # or standard

        all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)
        date_idx = np.where((all_dates >= start_date) &
                            (all_dates <= end_date))[0]

        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]

        nearest_lon = find_nearest(lons, lon)
        nearest_lat = find_nearest(lats, lat)
        lon_idx = np.where(lons == nearest_lon)[0]
        lat_idx = np.where(lats == nearest_lat)[0]
        
        df = pd.DataFrame([], index=all_dates[date_idx],
                          columns=params)

        for key in params:
            data = ncfile.variables[key][date_idx, lat_idx, lon_idx]
            param_data = np.array(data)
            
            df[key] = param_data.flatten()
            
    return df, nearest_lon, nearest_lat


def read_ts_area(path, param, lat_min, lat_max, lon_min, lon_max, t=1,
                 poets=False):
    '''
    Reads all pixel of given area and returns the mean value per day
    for this area.
    Parameters:
    -----------
    path : str
        Path to stacked netcdf file
    param : str
        Parameter to be read (name as in nc-files).
    lat/lon_min/max : float, optional
        Bounding box coordinates, area to be read
    t : int, optional
        T-value of SWI, default=1
    Returns:
    --------
    data : pd.DataFrame
        Dataset containing mean value of whole area for each date
    '''
    with Dataset(path, "r") as ncfile:
        unit_temps = ncfile.variables['time'].units
        nctime = ncfile.variables['time'][:]
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError:  # Attribute doesn't exist
            cal_temps = u"gregorian"  # or standard

        all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)

        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]
        if param == 'SWI':
            param = 'SWI_' + str(t).zfill(3)
        mean = []
        dates = []
        start_day = datetime(2007,1,1)
        for day in all_dates:
            if day < start_day: 
                continue
            nearest_date = find_nearest(all_dates, day)
            date_idx = np.where(all_dates == nearest_date)[0]

            lons = ncfile.variables['lon'][:]
            lats = ncfile.variables['lat'][:]
            lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
            lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
            data = ncfile.variables[param][date_idx, lat_idx, lon_idx]
            data[data < 0] = 0
            if poets:
                data = np.ma.masked_where(((data.data == -99)|(data.data==0)), data)

            if np.ma.is_masked(data):
                mean_value = data.mean()
            else:
                mean_value = data.mean()
            if np.ma.is_masked(mean_value):
                continue
            mean.append(mean_value)
            dates.append(day)

        data_df = {param: mean}
        df = pd.DataFrame(data=data_df, index=dates)
        if df.columns == 'SWI':
            df.columns = [param]

    return df


def find_nearest(array, element):
    idx = (np.abs(array - element)).argmin()
    return array[idx]