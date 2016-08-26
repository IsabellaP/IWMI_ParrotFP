import os
import fnmatch
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from datetime import datetime
from pygrids.warp5 import DGGv21CPv20
from rsdata.WARP.interface import WARP
from warp_data.interface import init_grid
from pygeogrids.grids import BasicGrid
import matplotlib.pyplot as plt


# Code for reading various datasets (LAI, NDVI...)
def data_reader(datasets, paths, img=False, ts=False):
    
    for ds in datasets:
        if ds == 'lc':
            lc = read_LC(paths['lc'])#, lat_min=5.9180, lat_max=35.5,lon_min=68, lon_max=97)
        if ds == 'ssm':
            #read_foxy_finn(paths['ssm'])
            ssm = read_WARP_dataset(paths['ssm'])
        if img == True:
            for year in range(2007, 2014):
                for month in range(1, 13):
                    read_img(paths[ds], ds, timestamp=datetime(year,month,1), plot_img=True)
        if ts == True:
            read_ts(paths[ds], ds, gpi=389821, plot_ts=True)        


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
    grid_info = {'grid_class': DGGv21CPv20, 
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

def read_img(path, param='NDVI', swi_key='SWI_020', lat_min=5.9180, 
             lat_max=9.8281, lon_min=79.6960, lon_max=81.8916, 
             timestamp=datetime(2010,7,1), plot_img=False):
    """
    Parameters:
    -----------
    path : str
        Path to nc-data
    param : str, optional
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
    folders = os.listdir(path)

    if param == 'NDVI300':
        for fname in sorted(folders):
            year = int(fname[8:12])
            month = int(fname[12:14])
            day = int(fname[14:16])
            timestamp_array.append(datetime(year, month, day))
            
    else: # NDVI, LAI, SWI
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
        key = swi_key
    elif param == 'NDVI300':
        key = 'NDVI'
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
        plt.matshow(param_data, fignum=False)
        plt.colorbar()
        plt.title(param+', '+str(nearest_date))
        #plt.show()
        plt.savefig("C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\"+
                     key+"_"+str(timestamp.date())+".png")
        plt.clf()

    return param_data


def read_ts(path, param='NDVI', lon=80.5, lat=6.81, gpi=None,
            start_date=datetime(2010,1,1), end_date=datetime(2010,3,31),
            plot_ts=False, swi_param='SWI_001'):
    """
    Parameters:
    -----------
    path : str
        Path to nc-data
    param : str, optional
        Parameter to be read (name as in nc-files). Default: NDVI
    lon, lat : float, optional
        Longitude and latitude of point of interest, default: point in Sri 
        Lanka. Either lon and lat or gpi must be provided
    gpi : int, optional
        Grid Point Index, default: None. Either gpi or lon and lat must be 
        provided. Given gpi overwrites lon and lat.
    start_date, end_date : datetime.datetime, optional
        Start and end timestamp of time series, default: January 2014
    swi_param : str, optional
        possible variables: SWI_001, 005, 010, 015, 020, 040, 060, 100
        
    Returns:
    --------
    data : dict
        Dataset
    """

    folders = os.listdir(path)
    timestamp_array = []
      
    if param == 'NDVI300':
        for fname in sorted(folders):
            year = int(fname[8:12])
            month = int(fname[12:14])
            day = int(fname[14:16])
            timestamp_array.append(datetime(year, month, day))
            
    else: # NDVI, LAI, SWI
        for fname in sorted(folders):
            year = int(fname[0:4])
            month = int(fname[4:6])
            day = int(fname[6:8])
            timestamp_array.append(datetime(year, month, day))
    
    timestamp_array = np.array(timestamp_array)
    date_idx = np.where((timestamp_array>=start_date) & 
                        (timestamp_array<=end_date))[0]
    
    folderlist = np.array(folders)[date_idx]
    
    # init grid for lonlat/gpi conversion
    grid_info = {'grid_class': DGGv21CPv20, 
                 'grid_filename': 'C:\\Users\\i.pfeil\\Documents\\'+
                 '0_IWMI_DATASETS\\ssm\\DGGv02.1_CPv02.nc'}
    grid = init_grid(grid_info)
    
    if gpi is not None:
        # overwrite lon, lat if gpi given
        lon, lat = grid.gpi2lonlat(gpi)
    
    param_data = []
    if param == 'SWI':
        key = swi_param
    elif param == 'NDVI300':
        key = 'NDVI'
    else:
        key = param
    
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
            param_data.append(ncfile.variables[key][lat_idx, lon_idx][0][0])
    
    param_data = np.array(param_data)
    df = pd.DataFrame(param_data, index=timestamp_array[date_idx], 
                      columns=[key])
    
    if plot_ts == True:
        df.plot()
        plt.title(param+', lon: '+str(nearest_lon)+', lat: '+str(nearest_lat))
        plt.show()
        
    return df


def read_poets_nc(poets_path, start_date, end_date, gpi=None, lon=None, 
                  lat=None):
    
    if gpi is not None:
        grid = init_poets_grid()
        lon, lat = grid.gpi2lonlat(gpi)
    
    with Dataset(poets_path, "r") as ncfile:
        unit_temps = ncfile.variables['time'].units
        nctime = ncfile.variables['time'][:]
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError : # Attribute doesn't exist
            cal_temps = u"gregorian" # or standard
    
        timestamp = num2date(nctime, units = unit_temps, calendar = cal_temps)
        date_idx = np.where((timestamp >= start_date) & 
                            (timestamp <= end_date))[0]
        
        # find nearest lonlat
        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]
        nearest_lon = find_nearest(lons, lon)
        nearest_lat = find_nearest(lats, lat)
        lon_idx = np.where(lons==nearest_lon)[0]
        lat_idx = np.where(lats==nearest_lat)[0]
        
        ndvi = ncfile.variables['NDVI_dataset'][date_idx, lat_idx, lon_idx]
        swi1 = ncfile.variables['SWI_SWI_001'][date_idx, lat_idx, lon_idx]
        swi2 = ncfile.variables['SWI_SWI_010'][date_idx, lat_idx, lon_idx]
        swi3 = ncfile.variables['SWI_SWI_020'][date_idx, lat_idx, lon_idx]
        swi4 = ncfile.variables['SWI_SWI_040'][date_idx, lat_idx, lon_idx]
        swi5 = ncfile.variables['SWI_SWI_060'][date_idx, lat_idx, lon_idx]
        swi6 = ncfile.variables['SWI_SWI_100'][date_idx, lat_idx, lon_idx]
    
    ndvi = np.vstack(ndvi)[:,0]
    ndvi[(ndvi==-99)] = np.NaN
    
    swi1 = np.vstack(swi1)[:,0]
    swi2 = np.vstack(swi2)[:,0]
    swi3 = np.vstack(swi3)[:,0]
    swi4 = np.vstack(swi4)[:,0]
    swi5 = np.vstack(swi5)[:,0]
    swi6 = np.vstack(swi6)[:,0]
    
    ndvi = pd.DataFrame(ndvi, columns=['NDVI'], index=timestamp[date_idx])
    swi1 = pd.DataFrame(swi1, columns=['SWI_001'], index=timestamp[date_idx])
    swi2 = pd.DataFrame(swi2, columns=['SWI_010'], index=timestamp[date_idx])
    swi3 = pd.DataFrame(swi3, columns=['SWI_020'], index=timestamp[date_idx])
    swi4 = pd.DataFrame(swi4, columns=['SWI_040'], index=timestamp[date_idx])
    swi5 = pd.DataFrame(swi5, columns=['SWI_060'], index=timestamp[date_idx])
    swi6 = pd.DataFrame(swi6, columns=['SWI_100'], index=timestamp[date_idx])

    return ndvi, swi1, swi2, swi3, swi4, swi5, swi6


def find_nearest(array, element):
    return min(array, key=lambda x: abs(x - element))


def init_SWI_grid(fpath=None):
    
    if fpath is None:
        fpath = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\SWI\\20070701"
    fname = "g2_BIOPAR_SWI10_200707010000_GLOBE_ASCAT_V3_0_1.nc"
    with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]
        mask = (ncfile.variables["SWI_010"][:]).mask

    lons, lats = np.meshgrid(lon, lat)
    grid = BasicGrid(lons[np.where(mask == False)], lats[np.where(mask == False)])
    
    return grid


def init_poets_grid(fpath=None):
    
    if fpath is None:
        fpath = "C:\\Users\\i.pfeil\\Desktop\\poets\\DATA\\"
    fname = "West_SA_0.4_dekad.nc"
    with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]
        mask = ncfile.variables['SWI_SWI_001'][0,:,:].mask

    lons, lats = np.meshgrid(lon, lat)
    grid = BasicGrid(lons[np.where(mask == False)], lats[np.where(mask == False)])
    
    return grid


if __name__ == '__main__':
    
    # read Sri Lanka gpis
    gpi_path = "C:\\Users\\i.pfeil\\Desktop\\Isabella\\pointlist_Sri Lanka_warp.csv"
    gpis_df = pd.read_csv(gpi_path)
    
    # set paths to datasets
    ssm_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ssm\\foxy_finn\\R1A\\"
    lcpath = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ESACCI-LC-L4-LCCS-Map-300m-P5Y-2010-v1.6.1.nc"
    ndvi_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\NDVI\\"
    ndvi300_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\NDVI300\\"
    lai_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\LAI\\"
    swi_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\SWI\\"
    fapar_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\FAPAR\\"
    poets_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\DATA\\West_SA_0.4_dekad.nc"
    
    #read_poets_nc(poets_path)
    
    datasets = ['NDVI']
    paths = {'ssm': ssm_path, 'lc': lcpath, 'NDVI300': ndvi300_path, 
             'NDVI': ndvi_path, 'LAI': lai_path, 'SWI': swi_path, 
             'FAPAR': fapar_path}
    
    read_poets_nc(poets_path, datetime(2010,1,1), datetime(2012,1,1), lon=75, 
                  lat=20)
    
    
    print 'done'