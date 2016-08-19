import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data_readers import read_ts 

import pytesmo.scaling as scaling
import pytesmo.temporal_matching as temp_match
import pytesmo.metrics as metrics


def plot_alltogether(time_lag, gpi, ts1, ts2, *args):

    matched_data = temp_match.matching(ts1, ts2, *args)
    if len(matched_data) == 0:
        print "Empty dataset."
        return
    #scaled_data = scaling.scale(matched_data, method="mean_std")
    matched_data.plot(figsize=(15, 5))
    plt.title('SWI and Vegetation indices comparison (rescaled)')
    #plt.show()
    plt.savefig("C:\\Users\\i.pfeil\\Desktop\\TS_plots\\"+str(gpi)+"_"+
                str(time_lag)+".png")
    plt.clf()


def rescale_peng(vi, vi_min, vi_max):
    
    vi_resc = (vi - vi_min) / (vi_max - vi_min) * 100
    return vi_resc


def corr(paths, gpi, corr_df, start_date, end_date, time_lag=0, plot_fig=False):
    
    swi_path = paths['SWI']
    lai_path = paths['LAI']
    #ndvi_path = paths['NDVI']
    #fapar_path = paths['FAPAR']
    
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
    
    lai = read_ts(lai_path, gpi=gpi, param='LAI', start_date=start_date,
                   end_date=end_date)
    #===========================================================================
    # ndvi = read_ts(ndvi_path, gpi=gpi, param='NDVI', start_date=start_date,
    #                end_date=end_date)
    # fapar = read_ts(fapar_path, gpi=gpi, param='FAPAR', start_date=start_date,
    #                end_date=end_date)
    #===========================================================================
    
    # rescale LAI before further processing
    lai = rescale_peng(lai, lai.min(), lai.max())
    print lai
    lai.plot()
    
    # insert time lag
    if time_lag > 0:
        lai_idx = lai.index + timedelta(days=time_lag)
        lai = pd.DataFrame(lai.values, columns=['LAI'], index=lai_idx)

    if plot_fig:
        print gpi
        plot_alltogether(time_lag, gpi, swi1, swi2, swi3, swi4, swi5, swi6, lai)
    
    water = {'SWI_001': swi1, 'SWI_010': swi2, 'SWI_020': swi3, 
             'SWI_040': swi4, 'SWI_060': swi5, 'SWI_100': swi6}
    
    vegetation = {'LAI': lai}#, 'NDVI': ndvi, 'FAPAR': fapar} 
    
    # plot lai time lags over SWI
    if time_lag == 0:
        lai0 = lai.copy()
        lai_idx10 = lai.index + timedelta(days=10)
        lai10 = pd.DataFrame(lai.values, columns=['LAI10'], index=lai_idx10)
        lai_idx20 = lai.index + timedelta(days=20)
        lai20 = pd.DataFrame(lai.values, columns=['LAI20'], index=lai_idx20)
        lai_idx30 = lai.index + timedelta(days=30)
        lai30 = pd.DataFrame(lai.values, columns=['LAI30'], index=lai_idx30)
        lai_idx40 = lai.index + timedelta(days=40)
        lai40 = pd.DataFrame(lai.values, columns=['LAI40'], index=lai_idx40)
        lai_idx50 = lai.index + timedelta(days=50)
        lai50 = pd.DataFrame(lai.values, columns=['LAI50'], index=lai_idx50)
        lai_idx60 = lai.index + timedelta(days=60)
        lai60 = pd.DataFrame(lai.values, columns=['LAI60'], index=lai_idx60)
                
        for w_key in water.keys():
            plot_alltogether(w_key, gpi, water[w_key], lai0, lai10, lai20, lai30, 
                             lai40, lai50, lai60)

    for ds_veg in vegetation.keys():
        for ds_water in sorted(water.keys()):
            data_together = temp_match.matching(water[ds_water], 
                                                vegetation[ds_veg])
            rho, p = metrics.spearmanr(data_together[ds_water], 
                                    data_together[ds_veg])
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
            

def zribi(paths, gpi, start_date, end_date, plot_fig=False, monthly=False):
    
    t_val = 'SWI_020'
    swi_path = paths['SWI']
    ndvi_path = paths['LAI']
    
    swi = read_ts(swi_path, gpi=gpi, param='SWI', start_date=start_date, 
                  end_date=end_date, swi_param=t_val)#[t_val]
    
    ndvi = read_ts(ndvi_path, gpi=gpi, param='LAI', start_date=start_date,
                   end_date=end_date)
    
    if monthly:
        swi = swi.resample("M").mean()
        ndvi = ndvi.resample("M").mean()
#===============================================================================
#     # simulate swi and ndvi time series to test simulation
#     Fs = 365
#     f = 5
#     sample = 365
#     x = np.arange(sample)
#     y = np.sin(2 * np.pi * f * x / Fs)
#     y2 = np.sin(2 * np.pi * f * x / Fs + 50)
# 
#     swi['SWI_020'] = y[:len(swi)]
#     ndvi['NDVI'] = y2[:len(ndvi)]
#===============================================================================
                   
    dndvi = np.ediff1d(ndvi, to_end=np.NaN)
    ndvi['D_NDVI'] = pd.Series(dndvi, index=ndvi.index)
    matched_data = temp_match.matching(swi, ndvi)
      
    # calculate parameters k and d only from 2007-2010 data
    if monthly:
        grouped_data = matched_data['2007':'2010'].groupby(matched_data['2007':'2010'].index.month)
    else:
        grouped_data = matched_data['2007':'2010'].groupby([matched_data['2007':'2010'].index.month, 
                                                        matched_data['2007':'2010'].index.day])
      
    kd = {}
    for key, _ in grouped_data:
        x = grouped_data[t_val].get_group(key)
        y = grouped_data['D_NDVI'].get_group(key)
        k, d = np.polyfit(x, y, 1)
        kd[key] = [k, d]
        if plot_fig:
            plt.plot(x, y, '*')
            plt.plot(np.arange(100), np.arange(100)*k+d, "r")
            plt.title('Month, Day: '+str(key)+', f(x) = '+str(round(k, 3))+
                      '*x + '+str(round(d, 3)))
            plt.xlabel(t_val)
            plt.ylabel('D_NDVI')
            plt.show()
      
    # simulation - integrate forecast length
    ndvi_sim = [ndvi['LAI'][0]]
    for i in range(1,len(matched_data)):
        print i, matched_data.index[i]
        try:
            if monthly:
                k, d = kd[matched_data.index[i].month]
            else:
                k, d = kd[(matched_data.index[i].month, matched_data.index[i].day)]
        except KeyError:
            ndvi_sim.append(ndvi_sim[i-1])
            if monthly:
                print 'no k, d values for month '+str(matched_data.index[i].month)
            else:
                print 'no k, d values for '+str((matched_data.index[i].month, matched_data.index[i].day))
            continue
          
        prev_date = (matched_data.index[i]-matched_data.index[i-1]).days
        if prev_date > 6: # days to latest available ndvi value
            ndvi_prev = np.NaN # NaN if latest ndvi value is older than prev_date
        else:
            # use ndvi instead of ndvi_sim to keep the forecast_length of 10 days
            ndvi_prev = matched_data['LAI'][i-1]
        ndvi_sim1 = ndvi_prev + k*matched_data[t_val][i] + d
        print ndvi_prev, k, d, ndvi_sim1
        ndvi_sim.append(ndvi_sim1)
      
    results = pd.DataFrame(matched_data[t_val].values, columns=[t_val],
                           index=matched_data.index)
    results['NDVI'] = pd.Series(matched_data['LAI'].values*100, 
                                index=matched_data.index)
    results['NDVI_sim'] = pd.Series(np.multiply(ndvi_sim, 100), 
                                    index=matched_data.index)
    results.plot()
    plt.title(str(gpi)+', t value: '+t_val)
    plt.show()
      
    return ndvi_sim


if __name__ == '__main__':
    
    # read Sri Lanka gpis
    #gpi_path = "C:\\Users\\i.pfeil\\Desktop\\Isabella\\pointlist_Sri Lanka_warp.csv"
    #gpis_df = pd.read_csv(gpi_path)
    
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
    
    #data_reader(datasets, paths)
    
    start_date = datetime(2008, 1, 1)
    end_date = datetime(2009, 1, 1)

    gpi = 1057207
    gpi = 1032891
    gpi = 1093603
    gpi = 1075401
    #zribi(paths, gpi, start_date, end_date, plot_fig=False, monthly=True)
    
    time_lags = [0, 10, 20, 30, 40, 50, 60]
    corr_df = pd.DataFrame([], index=time_lags)
    for time_lag in time_lags:
        print time_lag
        corr_df = corr(paths, gpi, corr_df, start_date, end_date, 
                       time_lag=time_lag, plot_fig=False)
    
    print corr_df
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
    
    print 'done'
