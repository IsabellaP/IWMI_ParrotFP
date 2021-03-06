import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from Basemap_scatterplot import scatterplot
import matplotlib.colors as cls

from readers import read_ts, read_img

import pytesmo.scaling as scaling
import pytesmo.temporal_matching as temp_match
import pytesmo.metrics as metrics
from pytesmo.grid.resample import resample_to_grid


def plot_alltogether(time_lag, lon, lat, ts1, ts2, scale_ts=False, 
                     save_fig=False, *args):

    matched_data = temp_match.matching(ts1, ts2, *args)
    if len(matched_data) == 0:
        print "Empty dataset."
        return
    if scale_ts:
        matched_data = scaling.scale(matched_data, method="mean_std")
    
    matched_data.plot(figsize=(15, 5))
    plt.title('SWI and Vegetation indices comparison (rescaled)')
    if save_fig:
        plt.savefig("C:\\Users\\i.pfeil\\Desktop\\TS_plots\\lon_"+str(lon)+
                    "_lat_"+str(lat)+'_'+str(time_lag)+".png", 
                    bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def rescale_peng(vi, vi_min, vi_max):
    
    vi_resc = (vi - vi_min) / (vi_max - vi_min) * 100
    return vi_resc


def corr(paths, corr_df, start_date, end_date, lon=None, lat=None, 
         vi_str='NDVI', time_lags=[0, 10, 20, 30, 40, 50, 60, 100], 
         plot_time_lags=False):

    """ Calculate Spearman's Rho and p-value for SWI (all t-values) and 
    specified VI (default NDVI).
    If plot_time_lags is True, a plot of VI (for different time lags) over
    SWI (all t-values) is created.
    
    Parameters:
    -----------
    paths : dict
        Paths to datasets
    corr_df : pd.DataFrame
        DataFrame where correlation coeff.s are stored
    start_date, end_date : datetime
        Start and end date
    vi_str : str, optional
        Vegetation index to use, default: NDVI
    time_lag : int, optional
        time lag for shifting VI, default: 0 (days)
    plot_time_lags : bool, optional
        Plot (shifted) VI over SWIs, default: False
        
    Returns:
    --------
    corr_df : pd.DataFrame
        DataFrame containing the correlation coeff.s
    """

    swi_path = paths['SWI']
    vi_path = paths[vi_str]
    
    # read SWI for different T-values and VI
    swi_list = ['SWI_001', 'SWI_010', 'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']
    swi_df = read_ts(swi_path, lon=lon, lat=lat, params=swi_list, 
                     start_date=start_date, end_date=end_date)
    vi = read_ts(vi_path, lon=lon, lat=lat, params=vi_str, 
                 start_date=start_date, end_date=end_date)
    vi[vi_str][np.where(vi==255)[0]] = np.NaN

    water = {}
    for swi_key in swi_list:
        water[swi_key] = swi_df[swi_key] 
    
    # rescale VI before further processing using method from Peng et al., 2014
    for ds_water in water:
        water[ds_water] = rescale_peng(water[ds_water], 
                                       np.nanmin(water[ds_water]), 
                                       np.nanmax(water[ds_water]))
    
    vi_resc = rescale_peng(vi, np.nanmin(vi), np.nanmax(vi))
    
    # insert time lag
    for time_lag in time_lags:
        if time_lag > 0:
            vi = vi_resc.copy()
            vi_idx = vi.index + timedelta(days=time_lag)
            vi = pd.DataFrame(vi.values, columns=[vi_str], index=vi_idx)
    
        # plot vi time lags over SWI
        if plot_time_lags and time_lag == 0:
            vi0 = vi.copy()
            vi_idx10 = vi.index + timedelta(days=10)
            vi10 = pd.DataFrame(vi.values, columns=['vi10'], index=vi_idx10)
            vi_idx20 = vi.index + timedelta(days=20)
            vi20 = pd.DataFrame(vi.values, columns=['vi20'], index=vi_idx20)
            vi_idx30 = vi.index + timedelta(days=30)
            vi30 = pd.DataFrame(vi.values, columns=['vi30'], index=vi_idx30)
            vi_idx40 = vi.index + timedelta(days=40)
            vi40 = pd.DataFrame(vi.values, columns=['vi40'], index=vi_idx40)
            vi_idx50 = vi.index + timedelta(days=50)
            vi50 = pd.DataFrame(vi.values, columns=['vi50'], index=vi_idx50)
            vi_idx60 = vi.index + timedelta(days=60)
            vi60 = pd.DataFrame(vi.values, columns=['vi60'], index=vi_idx60)
            vi_idx100 = vi.index + timedelta(days=100)
            vi100 = pd.DataFrame(vi.values, columns=['vi100'], index=vi_idx100)
            
            plot_alltogether(0, lon, lat, swi_df, vi0, save_fig=True)
            plot_alltogether(10, lon, lat, swi_df, vi10, save_fig=True)
            plot_alltogether(20, lon, lat, swi_df, vi20, save_fig=True)
            plot_alltogether(30, lon, lat, swi_df, vi30, save_fig=True)
            plot_alltogether(40, lon, lat, swi_df, vi40, save_fig=True)
            plot_alltogether(50, lon, lat, swi_df, vi50, save_fig=True)
            plot_alltogether(60, lon, lat, swi_df, vi60, save_fig=True)
            plot_alltogether(100, lon, lat, swi_df, vi100, save_fig=True)
    
        vegetation = {vi_str: vi}
    
        # calculate Spearman's Rho and p-value for VI and SWIs
        for ds_veg in vegetation.keys():
            for ds_water in sorted(water.keys()):
                data_together = temp_match.matching(water[ds_water], 
                                                    vegetation[ds_veg])
                rho, p = metrics.spearmanr(data_together[ds_water], 
                                           data_together[ds_veg])
                # mask values with p-value > 0.05
                if p > 0.05:
                    rho = np.NaN
                if ds_veg+'_'+ds_water+'_rho' in corr_df.columns:
                    corr_df[ds_veg+'_'+ds_water+'_rho'].iloc[np.where(corr_df.index==
                                                                  time_lag)] = rho
                    corr_df[ds_veg+'_'+ds_water+'_p'].iloc[np.where(corr_df.index==
                                                                time_lag)] = p
                else:
                    corr_df[ds_veg+'_'+ds_water+'_rho'] = pd.Series(rho, 
                                                                    index=[time_lag])
                    corr_df[ds_veg+'_'+ds_water+'_p'] = pd.Series(p, 
                                                                  index=[time_lag])
    
    return corr_df


def max_corr(corr_df, max_rho):
    
    for col in corr_df.columns:
        if 'rho' not in col:
            continue
        if col in max_rho.keys():
            max_rho[col].append(np.argmax(corr_df[col]))
        else:
            max_rho[col] = [np.argmax(corr_df[col])]
                
    return max_rho


def plot_corr(corr_df, lon, lat):
    
    """ Plot correlation values.
    
    Parameters:
    -----------
    corr_df : pd.DataFrame
        result from corr()
    """

    plot_cols = []
    for col in corr_df.columns:
        if 'rho' in col:
            plot_cols.append(col)
    
    plot_df = corr_df[plot_cols]
    plot_df.plot(style='o-')
    plt.title('Lon: '+str(lon)+', lat: '+str(lat))
    plt.xlabel("Time lag between datasets [days]")
    plt.ylabel("Spearman's Rho")
    plt.show()


def plot_rho(max_rho, lons, lats):
    
    for SWI_key in max_rho:
        fig = plt.figure(figsize=[20, 15])
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        m = Basemap(projection='cyl', ax=ax, llcrnrlat=10, urcrnrlat=35,
                    llcrnrlon=65, urcrnrlon=85)
        m.drawcoastlines()
        m.drawcountries()
        
        parallels = np.arange(-90,90,1.)
        m.drawparallels(parallels,labels=[1,0,0,0])
        meridians = np.arange(-180,180,1.)
        m.drawmeridians(meridians,labels=[0,0,0,1])
        
        # define the colormap
        cmap = plt.get_cmap('jet', 20)
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        
        # define the bins and normalize
        bounds = np.linspace(0,100,11)
        norm = cls.BoundaryNorm(bounds, cmap.N)
        
        sc = m.scatter(lons, lats, c=max_rho[SWI_key], edgecolor='None', s=210, marker=",", cmap=cmap, norm=norm)
        m.colorbar(sc, 'right', size='5%', pad='2%')
        plt.title('Time lag leading to highest correlation between VI and '+
                  SWI_key)
        #plt.show()
        plt.savefig('C:\\Users\\i.pfeil\\Desktop\\timelags\\'+SWI_key+'.png', bbox_inches='tight')
    
    print 'done'
    

def avg_gpis(param, path, start_date=datetime(2008,1,1), 
             end_date=datetime(2009,1,1), swi_key=None,
             lat_min=19.323746, lat_max=19.436668, 
             lon_min=74.870667, lon_max=74.938034):
    
    folderlist = os.listdir(path)
    mean_param = []
    dates_param = []
    dates_list = [datetime.strptime(date, '%Y%m%d') for date in folderlist]
    dates_array = np.array(dates_list)
    dates_idx = np.where((np.array(dates_list) > start_date) &
                         (np.array(dates_list) < end_date))[0]
    for day in dates_array[dates_idx]:
        param_data = read_img(path, param=param, swi_key=swi_key,
                              lat_min=lat_min - 0.05, 
                              lat_max=lat_max + 0.05,
                              lon_min=lon_min - 0.05, 
                              lon_max=lon_max + 0.05, 
                              timestamp=day, plot_img=False)
        if np.ma.is_masked(param_data):
            mean = param_data.data[np.where(param_data.data != 255)].mean()
        else:
            mean = param_data.mean()
        mean_param.append(mean)
        dates_param.append(day)
        
    if param == 'SWI':
        col_name = swi_key
    else:
        col_name = param
    df_avg = pd.DataFrame(data=mean_param, index=dates_param, 
                          columns=[col_name])
    return df_avg


def resample_rho_LC(input_data, src_lons, src_lats, target_lons, target_lats,
                    search_rad=18000):
    
    if src_lons.shape != src_lats.shape:
        src_lons, src_lats = np.meshgrid(src_lons, src_lats)
    
    if target_lons.shape != target_lats.shape:
        target_lons, target_lats = np.meshgrid(target_lons, target_lats)
    
    data_resamp = resample_to_grid({'data': input_data}, src_lons, src_lats, 
                                   target_lons, target_lats, 
                                   methods='nn', weight_funcs=None, 
                                   min_neighbours=1, 
                                   search_rad=search_rad, neighbours=8, 
                                   fill_values=None)

    return data_resamp['data']

def LC_mask(lons, lats, search_rad=80000):
    # read landcover classes
    lccs, lccs_lons, lccs_lats = read_img('C:\\Users\\i.pfeil\\Desktop\\'+
                                          'poets\\DATA\\West_SA_0.4_monthly_LC.nc',
                                          param='LC_lccs_class', lat_min=14.7, 
                                          lat_max=29.4, lon_min=68, 
                                          lon_max=81.8, 
                                          timestamp=datetime(2010, 1, 1))

    lccs_resamp = resample_rho_LC(lccs, lccs_lons, lccs_lats, lons, lats,
                                  search_rad=search_rad)
    
    #scatterplot(lons, lats, lccs_resamp, s=210, title='ESA CCI land cover classes, 0.4 deg.')
    
    no_data = (lccs_resamp == 0)
    urban = (lccs_resamp == -66)    
    water = (lccs_resamp == -46)
    snow_and_ice = (lccs_resamp == -36)
        
    mask_out = ((no_data) | (urban) | (water) | (snow_and_ice))
    lccs_masked = np.ma.masked_where(mask_out, lccs_resamp)
    
    return lccs_masked


def plot_max_timelags(lons, lats):
    
    max_corr_val = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_corr_val.npy')
    max_corr_swi = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_corr_swi.npy')
    max_corr_lag = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_corr_lag.npy')
    
    lccs_masked = LC_mask(lons, lats)
    max_corr_val = np.ma.masked_where(lccs_masked.mask, max_corr_val)
    max_corr_swi = np.ma.masked_where(lccs_masked.mask, max_corr_swi)
    max_corr_lag = np.ma.masked_where(lccs_masked.mask, max_corr_lag)
    
    scatterplot(lons, lats, max_corr_val, s=200, title='Maximum correlation '+
                'between SWI and NDVI', marker=',', 
                discrete=True, binmin=0, binmax=1, bins=20, 
                ticks=np.linspace(0,1,20))
    
    
    swi_vals = np.empty_like(max_corr_val)
     
    for idx, key in enumerate(np.unique(max_corr_swi)):
        print idx
        swi_vals[np.where(max_corr_swi == key)] = idx 
    scatterplot(lons, lats, swi_vals, s=200, title='SWI dataset showing '+
                'highest correlation with NDVI', marker=',', 
                discrete=True, binmin=0, binmax=7, bins=8, 
                ticks=np.unique(max_corr_swi))
    
    max_corr_lag[np.where(np.isnan(max_corr_lag))[0]] = np.NaN
    lag_vals = np.empty_like(max_corr_val)
    
    for idx, key in enumerate(np.unique(max_corr_lag)):
        print idx
        lag_vals[np.where(max_corr_lag == key)] = idx 
    scatterplot(lons, lats, lag_vals, s=200, title='Time lag showing '+
                'highest correlation between SWI and NDVI', marker=',', 
                discrete=True, binmin=0, binmax=8, bins=10, 
                ticks=np.unique(max_corr_lag))
    

def plot_corr_new(lons, lats):
    
    swi = {}
    swi['swi001'] = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\corr_swi001.npy')
    swi['swi010'] = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\corr_swi010.npy')
    swi['swi020'] = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\corr_swi020.npy')
    swi['swi040'] = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\corr_swi040.npy')
    swi['swi060'] = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\corr_swi060.npy')
    swi['swi100'] = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\corr_swi100.npy')

    lccs_masked = LC_mask(lons, lats)

    for key in swi:
        swi[key] = np.ma.masked_where(lccs_masked.mask, swi[key]) 
        scatterplot(lons, lats, swi[key], s=30, title='Correlation between '+
                    key+' and NDVI at a timelag of 3 days', marker=',', 
                    discrete=False, key=key)
