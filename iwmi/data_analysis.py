import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data_readers import read_ts, read_img
from Basemap_scatterplot import scatterplot

import pytesmo.scaling as scaling
import pytesmo.temporal_matching as temp_match
import pytesmo.metrics as metrics
from pygrids.warp5 import DGGv21CPv20
from warp_data.interface import init_grid


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


def corr(paths, gpi, corr_df, start_date, end_date, vi_str='NDVI', time_lag=0, 
         plot_ts=False, plot_time_lags=False, read_gpi_wise=False):

    """ Calculate Spearman's Rho and p-value for SWI (all t-values) and 
    specified VI (default NDVI).
    If plot_time_lags is True, a plot of VI (for different time lags) over
    SWI (all t-values) is created.
    
    Parameters:
    -----------
    paths : dict
        Paths to datasets
    gpi : int
        grid point (WARP grid, see http://rs.geo.tuwien.ac.at/dv/dgg/)
    corr_df : pd.DataFrame
        DataFrame where correlation coeff.s are stored
    start_date, end_date : datetime
        Start and end date
    vi_str : str, optional
        Vegetation index to use, default: NDVI
    time_lag : int, optional
        time lag for shifting VI, default: 0 (days)
    plot_ts : bool, optional
        Plot time series of all datasets, default: False
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
    if read_gpi_wise:
        swi1 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                       end_date=end_date, swi_param='SWI_001')
        swi2 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                       end_date=end_date, swi_param='SWI_010')
        swi3 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                       end_date=end_date, swi_param='SWI_020')
        swi4 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                       end_date=end_date, swi_param='SWI_040')
        swi5 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                       end_date=end_date, swi_param='SWI_060')
        swi6 = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                       end_date=end_date, swi_param='SWI_100')
        
        vi = read_ts(vi_path, gpi=gpi, param=vi_str, start_date=start_date,
                    end_date=end_date)
    
    else: # average gpis
        swi1 = avg_gpis('SWI', swi_path, start_date, end_date, swi_key='SWI_001')
        swi2 = avg_gpis('SWI', swi_path, start_date, end_date, swi_key='SWI_010')
        swi3 = avg_gpis('SWI', swi_path, start_date, end_date, swi_key='SWI_020')
        swi4 = avg_gpis('SWI', swi_path, start_date, end_date, swi_key='SWI_040')
        swi5 = avg_gpis('SWI', swi_path, start_date, end_date, swi_key='SWI_060')
        swi6 = avg_gpis('SWI', swi_path, start_date, end_date, swi_key='SWI_100')
        vi = avg_gpis(vi_str, vi_path, start_date, end_date)
    
    # rescale VI before further processing using method from Peng et al., 2014
    vi = rescale_peng(vi, vi.min(), vi.max())
    
    # insert time lag
    if time_lag > 0:
        vi_idx = vi.index + timedelta(days=time_lag)
        vi = pd.DataFrame(vi.values, columns=[vi_str], index=vi_idx)

    # plot datasets
    if plot_ts:
        print gpi
        plot_alltogether(time_lag, gpi, swi1, swi2, swi3, swi4, swi5, swi6, vi)
    
    water = {'SWI_001': swi1, 'SWI_010': swi2, 'SWI_020': swi3, 
             'SWI_040': swi4, 'SWI_060': swi5, 'SWI_100': swi6} 
    vegetation = {vi_str: vi} 
    
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
                
        for w_key in water.keys():
            plot_alltogether(w_key, gpi, water[w_key], vi0, vi10, vi20, vi30, 
                             vi40, vi50, vi60)

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


def plot_corr(corr_df, gpi):
    
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
    plt.title(str(gpi))
    plt.xlabel("Time lag between datasets [days]")
    plt.ylabel("Spearman's Rho")
    plt.show()


def plot_rho(gpis, max_rho):
    
    grid_info = {'grid_class': DGGv21CPv20, 
                 'grid_filename': 'C:\\Users\\i.pfeil\\Documents\\'+
                 '0_IWMI_DATASETS\\ssm\\DGGv02.1_CPv02.nc'}
    grid = init_grid(grid_info)
    lons, lats = grid.gpi2lonlat(gpis)
    
    for SWI_key in max_rho:
        scatterplot(lons, lats, max_rho[SWI_key], s=15, title=SWI_key)


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
    gpi_path = "C:\\Users\\i.pfeil\\Desktop\\Isabella\\pointlist_India_warp.csv"
    gpis_df = pd.read_csv(gpi_path)
    ind = np.where(gpis_df['cell']==1821)[0]
    gpis1821 = gpis_df['point'].values[ind]
    
    # set paths to datasets
    ssm_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ssm\\foxy_finn\\R1A\\"
    lcpath = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ESACCI-LC-L4-LCCS-Map-300m-P5Y-2010-v1.6.1.nc"
    ndvi_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\NDVI\\"
    ndvi300_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\NDVI300\\"
    lai_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\LAI\\"
    swi_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\SWI\\"
    fapar_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\FAPAR\\"
    
    paths = {'ssm': ssm_path, 'lc': lcpath, 'NDVI300': ndvi300_path, 
             'NDVI': ndvi_path, 'LAI': lai_path, 'SWI': swi_path, 
             'FAPAR': fapar_path}
    
    #zribi(paths, gpi, start_date, end_date, t_val='SWI_020', vi_str='NDVI',
    #      plot_fig=False, monthly=True)
     
    start_date = datetime(2007, 12, 20)
    end_date = datetime(2009, 1, 1)
    max_rho = {}
    for gp in sorted(gpis1821[1:10]):
        print gp
        time_lags = [0, 20, 40, 60]
        corr_df = pd.DataFrame([], index=time_lags)
        for time_lag in time_lags:
            print time_lag
            corr_df = corr(paths, gp, corr_df, start_date, end_date, 
                           vi_str='NDVI', time_lag=time_lag, plot_ts=False,
                           read_gpi_wise=True)
                           
            print corr_df
            
        #plot_corr(corr_df, gpi)
        max_rho = max_corr(corr_df, max_rho)
     
    # plot maps showing time lag with highest rho
    plot_rho(sorted(gpis1821[1:10]), max_rho)
    
    print 'done'
