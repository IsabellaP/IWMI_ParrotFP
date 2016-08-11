import os
import fnmatch
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from datetime import datetime
from pygrids.warp5 import DGGv21CPv20
from rsdata.WARP.interface import WARP
from warp_data.interface import init_grid
import matplotlib.pyplot as plt
import pytesmo.scaling as scaling
import pytesmo.temporal_matching as temp_match
import pytesmo.metrics as metrics


# Code for reading various datasets (LAI, NDVI...)
def data_reader(datasets, paths, img=False, ts=False):
    
    for ds in datasets:
        if ds == 'lc':
            lc = read_LC(paths['lc'])
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

def read_img(path, param='NDVI', lat_min=5.9180, lat_max=9.8281, 
            lon_min=79.6960, lon_max=81.8916, timestamp=datetime(2010,7,1), 
            plot_img=False):
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
        key = 'SWI_020'
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


def plot_alltogether(gpi, ts1, ts2, *args):

    matched_data = temp_match.matching(ts1, ts2, *args)
    if len(matched_data) == 0:
        print "Empty dataset."
        return
    scaled_data = scaling.scale(matched_data, method="mean_std")
    scaled_data.plot(figsize=(15, 5))
    plt.title('SWI and Vegetation indices comparison (rescaled)')
    #plt.show()
    plt.savefig("C:\\Users\\i.pfeil\\Desktop\\TS_plots\\"+str(gpi)+".png")
    plt.clf()


def corr(paths, gpi, start_date, end_date, plot_fig=False):
    
    swi_path = paths['SWI']
    lai_path = paths['LAI']
    ndvi_path = paths['NDVI']
    #fapar_path = paths['FAPAR']
    
    swi1 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                   end_date=end_date, swi_param='SWI_001')
    #===========================================================================
    # swi2 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
    #                end_date=end_date, swi_param='SWI_010')
    # swi3 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
    #                end_date=end_date, swi_param='SWI_020')
    # swi4 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
    #                end_date=end_date, swi_param='SWI_040')
    # swi5 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
    #                end_date=end_date, swi_param='SWI_060')
    #===========================================================================
    swi6 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                   end_date=end_date, swi_param='SWI_100')
    
    lai = read_ts(lai_path, gpi=gpi, param='LAI', start_date=start_date,
                   end_date=end_date)
    ndvi = read_ts(ndvi_path, gpi=gpi, param='NDVI', start_date=start_date,
                   end_date=end_date)
    #===========================================================================
    # fapar = read_ts(fapar_path, gpi=gpi, param='FAPAR', start_date=start_date,
    #                end_date=end_date)
    #===========================================================================

    if plot_fig:
        print gpi
        plot_alltogether(gpi, swi1, swi6, ndvi, lai)
    
    #===========================================================================
    # water = {'SWI_001': swi1, 'SWI_010': swi2, 'SWI_020': swi3, 
    #          'SWI_040': swi4, 'SWI_060': swi5, 'SWI_100': swi6}
    # vegetation = {'NDVI': ndvi, 'LAI': lai, 'FAPAR': fapar} 
    # 
    # print('gpi '+str(gpi))
    # print start_date, end_date
    # for ds_water in sorted(water.keys()):
    #     for ds_veg in vegetation.keys():
    #         data_together = temp_match.matching(water[ds_water], 
    #                                             vegetation[ds_veg])
    #         rho = metrics.spearmanr(data_together[ds_water], 
    #                                 data_together[ds_veg])
    #         print ds_water, ds_veg, rho
    #===========================================================================
            

def find_nearest(array, element):
    return min(array, key=lambda x: abs(x - element))


def zribi(paths, gpi, start_date, end_date, plot_fig=False):
    
    swi_path = paths['SWI']
    ndvi_path = paths['NDVI']
    
    swi = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                  end_date=end_date, swi_param='SWI_100')['SWI_100']
    
    ndvi = read_ts(ndvi_path, gpi=gpi, param='NDVI', start_date=start_date,
                   end_date=end_date)
                   
    dndvi = np.ediff1d(ndvi, to_end=np.NaN)
    ndvi['D_NDVI'] = pd.Series(dndvi, index=ndvi.index)
    matched_data = temp_match.matching(swi, ndvi)
    
    grouped_data = matched_data.groupby([matched_data.index.month, 
                                         matched_data.index.day])
    
    kd = {}
    for key, _ in grouped_data:
        x = grouped_data['SWI_100'].get_group(key)
        y = grouped_data['D_NDVI'].get_group(key)
        k, d = np.polyfit(x, y, 1)
        kd[key] = [k, d]
        if plot_fig:
            plt.plot(x, y, '*')
            plt.plot(np.arange(100), np.arange(100)*k+d, "r")
            plt.title('Month, Day: '+str(key)+', f(x) = '+str(round(k, 3))+
                      '*x + '+str(round(d, 3)))
            plt.xlabel('SWI_100')
            plt.ylabel('D_NDVI')
            plt.show()
    
    # simulation
    ndvi_sim = [ndvi['NDVI'][0]]
    for i in range(1,len(matched_data)):
        # stimmt index i-1
        k, d = kd[(matched_data.index[i].month, matched_data.index[i].day)]
        ndvi_sim.append(ndvi_sim[i-1] + k*matched_data['SWI_100'][i] + d)
    
    results = pd.DataFrame(matched_data['SWI_100'].values, columns=['SWI_100'],
                           index=matched_data.index)
    results['NDVI'] = pd.Series(matched_data['NDVI'].values*100, 
                                index=matched_data.index)
    results['NDVI_sim'] = pd.Series(np.multiply(ndvi_sim, 100), 
                                    index=matched_data.index)
    results.plot()
    plt.show()
    
    return ndvi_sim


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
    
    datasets = ['FAPAR']
    paths = {'ssm': ssm_path, 'lc': lcpath, 'NDVI300': ndvi300_path, 
             'NDVI': ndvi_path, 'LAI': lai_path, 'SWI': swi_path, 
             'FAPAR': fapar_path}
    
    #data_reader(datasets, paths, img=True, ts=False)
    
    start_date = datetime(2007, 1, 1)
    end_date = datetime(2014, 1, 1)
    #for gpi in gpis_df['point']:
    gpi = 542129
    zribi(paths, gpi, start_date, end_date, plot_fig=False)
    #corr(paths, gpi, start_date, end_date, plot_fig=True)
       
    print 'done'