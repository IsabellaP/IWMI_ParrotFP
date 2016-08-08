import os
import fnmatch
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from datetime import datetime
from pygrids.warp5 import DGGv21CPv20_ind_ld
from rsdata.WARP.interface import WARP
from warp_data.interface import init_grid
import matplotlib.pyplot as plt


# Code for reading various datasets (LAI, NDVI...)
def data_reader(datasets, paths, img=True, ts=False):
    
    for ds in datasets:
        if img == True:
            read_img(paths[ds], ds, plot_img=True)
        if ts == True:
            pass
            #read_ts()
    
    #===========================================================================
    # if 'NDVI_img' in tasks:    
    #     NDVI_img = read_img(paths['NDVI'], timestamp=datetime(2014,1,10), 
    #                         plot_img=True)
    # if 'NDVI' in tasks:
    #     # if HDF Error: run again
    #     NDVI = read_NDVI(paths['NDVI'], start_date=datetime(2014, 1, 1), 
    #                      end_date=datetime(2015, 02, 28), plot_ts=True)
    # if 'NDVI300' in tasks:
    #     # if HDF Error: run again
    #     NDVI300 = read_NDVI300(paths['NDVI300'], start_date=datetime(2014, 1, 1), 
    #                            end_date=datetime(2015, 02, 28), plot_ts=True)
    # if 'lc' in tasks:
    #     lc = read_LC(paths['lc'])
    # if 'ssm' in tasks:
    #     #read_foxy_finn(paths['ssm'])
    #     ssm = read_WARP_dataset(paths['ssm'])
    #===========================================================================


def read_foxy_finn(ssm_path):
    """Read ASCAT ssm, version: Foxy Finn. Only use if standard reader is not
    enough (read_WARP_dataset)"""
    
    cell = 30
    fname = str(cell).zfill(4)+'.nc'
    
    with Dataset(os.path.join(ssm_path, fname), mode='r') as ncfile:
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]
        sm = ncfile.variables['sm'][:]
        row_size = ncfile.variables['row_size'][:]
        nctime = ncfile.variables['time'][:] # get values
        unit_temps = ncfile.variables['time'].units
        
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError : # Attribute doesn't exist
            cal_temps = u"gregorian" # or standard
    
    timestamp = num2date(nctime,units = unit_temps,calendar = cal_temps)
    
    return sm, timestamp

def read_WARP_dataset(cfg_path):
    """Read WARP soil moisture
    """
    grid_info = {'grid_class': DGGv21CPv20_ind_ld, 
                 'grid_filename': 'C:\\Users\\i.pfeil\\Documents\\'+
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
        lat_idx = np.where((lat>=lat_min)&(lat<=lat_max))[0]
        lon_idx = np.where((lon>=lon_min)&(lon<=lon_max))[0]
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
    
    mask_out = ((no_data)|(urban)|(water)|(snow_and_ice))
    lccs_masked = np.ma.masked_where(mask_out, lccs)

    plt.imshow(lccs_masked)
    plt.colorbar()
    plt.title('ESA CCI land cover')
    plt.show()
    
    return lccs_masked

def read_img(path, param='NDVI', lat_min=5.9180, lat_max=9.8281, 
            lon_min=79.6960, lon_max=81.8916, timestamp=datetime(2014,1,1), 
            plot_img=False):
    """
    Parameters:
    -----------
    path : str
        Path to nc-data
    params : str, optional
        Parameter to be read (name as in nc-files). Default: NDVI
    lat/lon_min/max : float, optional
        Bounding box coordinates, default: Sri Lanka
    timestamp : datetime.datetime, optional
        Timestamp of image, default: 01/01/2014
    plot_img : bool, optional
        If true, result image is plotted, default: False
        
    Returns:
    --------
    data : dict
        Dataset
    """

    timestamp_array = []

    if param == 'NDVI300':
        folders = os.listdir(path)
        for fname in sorted(folders):
            year = int(fname[8:12])
            month = int(fname[12:14])
            day = int(fname[14:16])
            timestamp_array.append(datetime(year, month, day))
            
    else: # NDVI, LAI, SWI
        folders = os.listdir(path)
        for fname in sorted(folders):
            year = int(fname[0:4])
            month = int(fname[4:6])
            day = int(fname[6:8])
            timestamp_array.append(datetime(year, month, day))
            
    timestamp_array = np.array(timestamp_array)
    # find nearest timestamp
    nearest_date = find_nearest(timestamp_array, timestamp)
    date_idx = np.where(timestamp_array==nearest_date)[0]
    
    folder = np.array(sorted(folders))[date_idx][0]
    fpath = os.path.join(path, folder)
    fname = fnmatch.filter(os.listdir(fpath), '*.nc')[0]
    
    if param == 'SWI':
        # possible variables: SWI_001, 005, 010, 015, 020, 040, 060, 100
        key = 'SWI_020'
    else:
        key = param
    
    with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]
        
        lat_idx = np.where((lat>=lat_min)&(lat<=lat_max))[0]
        lon_idx = np.where((lon>=lon_min)&(lon<=lon_max))[0]
        param_data = ncfile.variables[key][lat_idx, lon_idx]
        
    if plot_img == True:
        plt.figure()
        plt.imshow(param_data)
        plt.colorbar()
        plt.title(param+', '+str(nearest_date))
        plt.show()

    return param_data


def read_NDVI300(path, params='NDVI', lon=80.5, lat=6.81, 
                 start_date=datetime(2014,1,1), end_date=datetime(2014,1,31),
                 plot_ts=False):
    """
    Parameters:
    -----------
    path : str
        Path to nc-data
    params : list, optional
        List of parameters to be read. Default: NDVI
    lon, lat : float, optional
        Longitude and latitude of point of interest, default: point in Sri Lanka
    start_date, end_date : datetime.datetime, optional
        Start and end timestamp of time series, default: January 2014
        
    Returns:
    --------
    data : dict
        Dataset
    """

    folders = os.listdir(path)
    timestamp = []
    for fname in sorted(folders):
        year = int(fname[8:12])
        month = int(fname[12:14])
        day = int(fname[14:16])
        timestamp.append(datetime(year, month, day))
    
    timestamp = np.array(timestamp)
    date_idx = np.where((timestamp>=start_date) & (timestamp<=end_date))[0]
    
    folderlist = np.array(folders)[date_idx]
    NDVI = []
    for folder in sorted(folderlist):
        fpath = os.path.join(path, folder)
        fname = fnmatch.filter(os.listdir(fpath), '*.nc')[0]
        with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
            lons = ncfile.variables['lon'][:]
            lats = ncfile.variables['lat'][:]
            
            # find nearest lonlat
            nearest_lon = find_nearest(lons, lon)
            nearest_lat = find_nearest(lats, lat)
            lon_idx = np.where(lons==nearest_lon)[0]
            lat_idx = np.where(lats==nearest_lat)[0]
            NDVI.append(ncfile.variables['NDVI'][lat_idx, lon_idx][0][0])
    
    NDVI = np.array(NDVI)
    df = pd.DataFrame(NDVI, index=timestamp[date_idx], columns=['NDVI'])
    
    if plot_ts == True:
        df.plot()
        plt.title(params+', lon: '+str(nearest_lon)+', lat: '+str(nearest_lat))
        plt.show()
        
    return df


def find_nearest(array, element):
    return min(array, key=lambda x: abs(x - element))


if __name__ == '__main__':
    
    ssm_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ssm\\foxy_finn\\R1A\\"
    lcpath = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ESACCI-LC-L4-LCCS-Map-300m-P5Y-2010-v1.6.1.nc"
    ndvi_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\NDVI\\"
    ndvi300_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\NDVI300\\"
    lai_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\LAI\\"
    swi_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\SWI\\"
    
    
    datasets = ['SWI']
    paths = {'ssm': ssm_path, 'lc': lcpath, 'NDVI300': ndvi300_path, 
             'NDVI': ndvi_path, 'LAI': lai_path, 'SWI': swi_path}
    
    data_reader(datasets, paths, img=True, ts=False)
    
    print 'done'