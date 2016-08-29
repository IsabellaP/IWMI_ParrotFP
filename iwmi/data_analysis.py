import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

from readers import read_ts, read_img
from data_readers import read_poets_nc, init_SWI_grid, init_poets_grid
from Basemap_scatterplot import scatterplot

import pytesmo.scaling as scaling
import pytesmo.temporal_matching as temp_match
import pytesmo.metrics as metrics


def plot_alltogether(time_lag, gpi, ts1, ts2, scale_ts=False, save_fig=False,
                     *args):

    matched_data = temp_match.matching(ts1, ts2, *args)
    if len(matched_data) == 0:
        print "Empty dataset."
        return
    if scale_ts:
        matched_data = scaling.scale(matched_data, method="mean_std")
    
    matched_data.plot(figsize=(15, 5))
    plt.title('SWI and Vegetation indices comparison (rescaled)')
    if save_fig:
        plt.savefig("C:\\Users\\i.pfeil\\Desktop\\TS_plots\\"+str(gpi)+"_"+
                str(time_lag)+".png")
        plt.clf()
    else:
        plt.show()


def rescale_peng(vi, vi_min, vi_max):
    
    vi_resc = (vi - vi_min) / (vi_max - vi_min) * 100
    return vi_resc


def corr(paths, corr_df, start_date, end_date, lon=None, lat=None, 
         vi_str='NDVI', time_lag=0, plot_time_lags=False):

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

    water = {}
    for swi_key in swi_list:
        water[swi_key] = swi_df[swi_key]
    
    vegetation = {vi_str: vi} 
    
    # rescale VI before further processing using method from Peng et al., 2014
    for ds_water in water:
        water[ds_water] = rescale_peng(water[ds_water], 
                                       np.nanmin(water[ds_water]), 
                                       np.nanmax(water[ds_water]))
    
    vi = rescale_peng(vi, np.nanmin(vi), np.nanmax(vi))
    
    # insert time lag
    if time_lag > 0:
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

    # calculate Spearman's Rho and p-value for VI and SWIs
    for ds_veg in vegetation.keys():
        for ds_water in sorted(water.keys()):
            data_together = temp_match.matching(water[ds_water], 
                                                vegetation[ds_veg])
            rho, p = metrics.spearmanr(data_together[ds_water], 
                                       data_together[ds_veg])
            # mask values with p-value > 0.05
            #if p > 0.05:
            #    rho = np.NaN
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
       
    #===========================================================================
    # for SWI_key in max_rho:
    #     scatterplot(lons, lats, max_rho[SWI_key], s=75, title=SWI_key)
    #===========================================================================
    
    for SWI_key in max_rho:
        fig = plt.figure()
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        m = Basemap(projection='cyl', ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        
        parallels = np.arange(-90,90,15.)
        m.drawparallels(parallels,labels=[1,0,0,0])
        meridians = np.arange(-180,180,15.)
        m.drawmeridians(meridians,labels=[0,0,0,1])
        
        #sc = m.scatter(lons, lats, c=max_rho[SWI_key], edgecolor='None', s=10)
        sc = m.scatter(lons, lats, c=max_rho[SWI_key], edgecolor='None', s=75, marker=',')
        m.colorbar(sc, 'right', size='5%', pad='2%')
        plt.title('Time lag leading to highest correlation between VI and '+
                  SWI_key)
        plt.show()
    
    print 'done'


def zribi(paths, gpi, start_date, end_date, t_val='SWI_020', vi_str='NDVI', 
          plot_fig=False, monthly=False):
    
    """ Simulate VI from SWI as in Zribi et al., 2010.
    
    Parameters:
    -----------
    paths : dict
        Paths to datasets
    gpi : int
        grid point (WARP grid, see http://rs.geo.tuwien.ac.at/dv/dgg/)
    start_date, end_date : datetime
        Start and end date
    t_val : str, optional
        T value, default: 20
    vi_str : str, optional
        Vegetation index to use, default: NDVI
    plot_fig : bool, optional
        If true, functions from which k and d are derived are plotted, 
        default: False
    monthly : bool, optional
        If true, data is resampled monthly, default: False
    
    Returns:
    --------
    vi_sim : np.array
        Array containing simulated VI
    """
    
    swi_path = paths['SWI']
    vi_path = paths[vi_str]
    
    # read SWI with given t value and VI
    swi = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                  end_date=end_date, swi_param=t_val)#[t_val]
    
    vi = read_ts(vi_path, gpi=gpi, param=vi_str, start_date=start_date,
                end_date=end_date)
    
    # resample monthly
    if monthly:
        swi = swi.resample("M").mean()
        vi = vi.resample("M").mean()

    # calculate differences between VIs of consecutive months
    dvi = np.ediff1d(vi, to_end=np.NaN)
    vi['D_VI'] = pd.Series(dvi, index=vi.index)
    matched_data = temp_match.matching(swi, vi)
      
    # calculate parameters k and d only from 2007-2010 data
    sim_start = '2007'
    sim_end = '2010'
    if monthly:
        grouped_data = matched_data[sim_start:
                                    sim_end].groupby(matched_data[sim_start:
                                                    sim_end].index.month)
    else:
        grouped_data = matched_data[sim_start:
                                    sim_end].groupby([matched_data[sim_start:
                                                    sim_end].index.month, 
                                                    matched_data[sim_start:
                                                    sim_end].index.day])
      
    kd = {}
    for key, _ in grouped_data:
        x = grouped_data[t_val].get_group(key)
        y = grouped_data['D_VI'].get_group(key)
        k, d = np.polyfit(x, y, 1)
        kd[key] = [k, d]
        if plot_fig:
            plt.plot(x, y, '*')
            plt.plot(np.arange(100), np.arange(100)*k+d, "r")
            plt.title('Month, Day: '+str(key)+', f(x) = '+str(round(k, 3))+
                      '*x + '+str(round(d, 3)))
            plt.xlabel(t_val)
            plt.ylabel('D_VI')
            plt.show()
      
    # simulation - integrate forecast length
    vi_sim = [vi[vi_str][0]]
    for i in range(1,len(matched_data)):
        print i, matched_data.index[i]
        try:
            if monthly:
                k, d = kd[matched_data.index[i].month]
            else:
                k, d = kd[(matched_data.index[i].month,
                           matched_data.index[i].day)]
        except KeyError:
            vi_sim.append(vi_sim[i-1])
            if monthly:
                print 'no k, d values for month '+str(matched_data.index[i].month)
            else:
                print 'no k, d values for '+str((matched_data.index[i].month,
                                                 matched_data.index[i].day))
            continue
          
        prev_date = (matched_data.index[i]-matched_data.index[i-1]).days
        if monthly:
            prev_lim = 60
        else:
            prev_lim = 20
        if prev_date > prev_lim: # days to latest available vi value
            vi_prev = np.NaN # NaN if latest vi value is older than prev_date
        else:
            # use vi instead of vi_sim to keep the forecast length of 10 days
            vi_prev = matched_data[vi_str][i-1]
        vi_sim1 = vi_prev + k*matched_data[t_val][i] + d
        print vi_prev, k, d, vi_sim1
        vi_sim.append(vi_sim1)
    
    # plot results
    results = pd.DataFrame(matched_data[t_val].values, columns=[t_val],
                           index=matched_data.index)
    results[vi_str] = pd.Series(matched_data[vi_str].values*100, 
                                index=matched_data.index)
    results[vi_str+'_sim'] = pd.Series(np.multiply(vi_sim, 100), 
                                    index=matched_data.index)
    results.plot()
    plt.title(str(gpi)+', t value: '+t_val)
    plt.show()
      
    return vi_sim


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


if __name__ == '__main__':
    
    # read Sri Lanka gpis
    #gpi_path = "C:\\Users\\i.pfeil\\Desktop\\Isabella\\pointlist_India_warp.csv"
    #gpis_df = pd.read_csv(gpi_path)
    #ind = np.where(gpis_df['cell']==1821)[0]
    #gpis1821 = gpis_df['point'].values[ind]
    
    # set paths to datasets
    ssm_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ssm\\foxy_finn\\R1A\\"
    lcpath = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ESACCI-LC-L4-LCCS-Map-300m-P5Y-2010-v1.6.1.nc"
    ndvi_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\NDVI_stack.nc"
    ndvi300_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\NDVI300\\"
    lai_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\LAI\\"
    swi_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\SWI_stack.nc"
    fapar_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\FAPAR\\"
    
    paths = {'ssm': ssm_path, 'lc': lcpath, 'NDVI300': ndvi300_path, 
             'NDVI': ndvi_path, 'LAI': lai_path, 'SWI': swi_path, 
             'FAPAR': fapar_path}
    
    #zribi(paths, gpi, start_date, end_date, t_val='SWI_020', vi_str='NDVI',
    #      plot_fig=False, monthly=True)
    
    #===========================================================================
    # grid = init_SWI_grid()
    # lons = grid.activearrlon
    # lats = grid.activearrlat
    # 
    # # shpfile-bbox
    # lonlat_idx = np.where((lats>=14.7) & (lats<=29.4) & (lons>=68.0) & 
    #                       (lons<=81.8))[0]
    # lons_shp = lons[lonlat_idx]
    # lats_shp = lats[lonlat_idx]
    #===========================================================================
    
    # poets lonlat
    grid = init_poets_grid()
    gpis, lons, lats = grid.get_grid_points()
    
    start_date = datetime(2007, 12, 20)
    end_date = datetime(2009, 1, 1)
    max_rho = {}
    time_lags = [0, 10, 20, 30, 40, 50, 60, 100]
    corr_df = pd.DataFrame([], index=time_lags)
    
    for i in range(len(gpis)):
        print i
        for time_lag in time_lags:
            #print time_lag
            corr_df = corr(paths, corr_df, start_date, end_date, lon=lons[i], 
                           lat=lats[i], vi_str='NDVI', time_lag=time_lag)
                           
        #corr_df.to_csv("C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\corr_"+
        #               str(lons[i])+'_'+str(lats[i])+'.csv')
                
            #plot_corr(corr_df, gpi)
        max_rho = max_corr(corr_df, max_rho)
     
    # plot maps showing time lag with highest rho
    max_rho_data = np.array([0, 100, 100, 100, 0, 10, 0, 20, 0, 10, 20, 30, 0, 0, 40, 20, 20, 
                             10, 10, 20, 0, np.NaN, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 
                             100, 100, 20, np.NaN, 0, 0, 10, 50, 50, 30, 0, 10, 30, 0, 30])
    max_rho = {'key': max_rho_data}
    plot_rho(max_rho, lons[:50], lats[:50])
    
    print 'done'
