import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from datetime import datetime
from pygrids.warp5 import DGGv21CPv20
from rsdata.WARP.interface import WARP
from warp_data.interface import init_grid
import matplotlib.pyplot as plt


def read_foxy_finn(ssm_path):
    """Read ASCAT ssm, version: Foxy Finn. Only use if standard reader is not
    enough (read_WARP_dataset)"""

    cell = 30
    fname = str(cell).zfill(4) + '.nc'

    with Dataset(os.path.join(ssm_path, fname), mode='r') as ncfile:
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]
        sm = ncfile.variables['sm'][:]
        row_size = ncfile.variables['row_size'][:]
        nctime = ncfile.variables['time'][:]  # get values
        unit_temps = ncfile.variables['time'].units

        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError:  # Attribute doesn't exist
            cal_temps = u"gregorian"  # or standard

    timestamp = num2date(nctime, units=unit_temps, calendar=cal_temps)

    return sm, timestamp


def read_WARP_dataset(cfg_path):
    """Read WARP soil moisture
    """
    grid_info = {'grid_class': DGGv21CPv20,
                 'grid_filename': 'C:\\Users\\s.hochstoger\\Desktop\\' +
                                  '0_IWMI_DATASETS\\ssm\\DGGv02.1_CPv02.nc'}
    grid = init_grid(grid_info)
    WARP_io = WARP(version='IRMA1_WARP56_P2R1', parameter='ssm_userformat',
                   cfg_path=cfg_path, grid=grid)

    gpi = 389821
    df = WARP_io.read(gpi)

    return df


def read_LC(path, lat_min=5.9180, lat_max=9.8281,
            lon_min=79.6960, lon_max=81.8916):
    """Read ESA CCI land cover.

    Parameters:
    -----------
    path : str
        Path to nc-file
    lat/lon_min/max : float, optional
        Bounding box coordinates, default: Sri Lanka
    """

    with Dataset(path, mode='r') as ncfile:
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        lccs = ncfile.variables['lccs_class'][lat_idx, lon_idx]

    no_data = (lccs == 0)
    cropland_rainfed = (lccs == 10)
    cropland_rainfed_herbaceous_cover = (lccs == 11)
    cropland_rainfed_tree_or_shrub_cover = (lccs == 12)
    cropland_irrigated = (lccs == 20)
    mosaic_cropland = (lccs == 30)
    mosaic_natural_vegetation = (lccs == 40)
    tree_broadleaved_evergreen_closed_to_open = (lccs == 50)
    tree_broadleaved_deciduous_closed_to_open = (lccs == 60)
    tree_broadleaved_deciduous_closed = (lccs == 61)
    tree_broadleaved_evergreen_open = (lccs == 62)
    tree_needleleaved_evergreen_closed_to_open = (lccs == 70)
    tree_needleleaved_evergreen_closed = (lccs == 71)
    tree_needleleaved_evergreen_open = (lccs == 72)
    tree_needleleaved_deciduous_closed_to_open = (lccs == 80)
    tree_needleleaved_deciduous_closed = (lccs == 81)
    tree_needleleaved_deciduous_open = (lccs == 82)
    tree_mixed = (lccs == 90)
    mosaic_tree_and_shrub = (lccs == 100)
    mosaic_herbaceous = (lccs == 110)
    shrubland = (lccs == 120)
    shrubland_evergreen = (lccs == 121)
    shrubland_deciduous = (lccs == 122)
    grassland = (lccs == -126)
    lichens_and_mosses = (lccs == -116)
    sparse_vegetation = (lccs == -106)
    sparse_shrub = (lccs == -104)
    sparse_herbaceous = (lccs == -103)
    tree_cover_flooded_fresh_or_brakish_water = (lccs == -96)
    tree_cover_flooded_saline_water = (lccs == -86)
    shrub_or_herbaceous_cover_flooded = (lccs == -76)
    urban = (lccs == -66)
    bare_areas = (lccs == -56)
    bare_areas_consolidated = (lccs == -55)
    bare_areas_unconsolidated = (lccs == -54)
    water = (lccs == -46)
    snow_and_ice = (lccs == -36)

    mask_out = ((no_data) | (urban) | (water) | (snow_and_ice))
    lccs_masked = np.ma.masked_where(mask_out, lccs)

    plt.imshow(lccs_masked)
    plt.colorbar()
    plt.title('ESA CCI land cover')
    plt.show()

    return lccs_masked


def read_img(path, param='SWI_020', lat_min=5.9180, lat_max=9.8281,
             lon_min=79.6960, lon_max=81.8916, timestamp=datetime(2010, 7, 1)):
    """
    Parameters:
    -----------
    path : str
        Path to nc-data
    param : str
        Parameter to be read (name as in nc-files). Default: SWI_020
    lat/lon_min/max : float
        Bounding box coordinates, default: Sri Lanka
    timestamp : datetime.datetime
        Timestamp of image, default: 01/01/2014

    Returns:
    --------
    data : np.ma array
        Dataset
    """
    key = param
    with Dataset(path, "r") as ncfile:
        unit_temps = ncfile.variables['time'].units
        nctime = ncfile.variables['time'][:]
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError:  # Attribute doesn't exist
            cal_temps = u"gregorian"  # or standard

        all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)
        nearest_date = find_nearest(all_dates, timestamp)
        date_idx = np.where(all_dates == nearest_date)[0]

        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]
        lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
        data = ncfile.variables[key][date_idx, lat_idx, lon_idx]

    return data


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
            
    return df


def find_nearest(array, element):
    idx = (np.abs(array - element)).argmin()
    return array[idx]


pass