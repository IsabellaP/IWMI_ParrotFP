import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pytesmo.temporal_matching as temp_match
from poets.shape.shapes import Shape

from Basemap_scatterplot import scatter_subplots
from data_analysis import rescale_peng
from readers import read_ts, read_img, find_nearest, read_AG_LC


def validate_prediction(pred, vi_path, plotname):

    lons = pred[:,0]
    lats = pred[:,1]
    data = pred[:,2]
    timestamp = np.unique(pred[:,3])[0]
    
    # districts
    shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                             '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                             'IND_adm2')
    
    shpfile = Shape(region, shapefile=shapefile)
    lon_min, lat_min, lon_max, lat_max = shpfile.bbox
    
    vi_ts = datetime(timestamp.year, timestamp.month, timestamp.day)
    vi_data, vi_lons, vi_lats, vi_date = read_img(vi_path, param='NDVI', 
                                                  lat_min=lat_min, lat_max=lat_max,
                                                  lon_min=lon_min, lon_max=lon_max, 
                                                  timestamp=vi_ts)

    vi_data = np.ma.masked_where(vi_data<0, vi_data)

    vi_lon, vi_lat = np.meshgrid(vi_lons, vi_lats)
    vi_lon = vi_lon.flatten()
    vi_lat = vi_lat.flatten()
    
    scatter_subplots(lons, lats, data, 250, 
                     vi_lon, vi_lat, vi_data, 1, plotname,
                     llcrnrlat=lat_min-0.5, urcrnrlat=lat_max+0.5,
                     llcrnrlon=lon_min-0.5, urcrnrlon=lon_max+0.5,
                     vi_date=vi_date)
    
    return pred

def start_pred(paths, region, vi_str='NDVI', t_val='SWI_040',
               mode='calc_kd'):
    
    with Dataset(paths['SWI'], 'r') as ncfile:
        res_lons = ncfile.variables['lon'][:]
        res_lats = ncfile.variables['lat'][:]
        
    lccs, lc_lons, lc_lats = read_AG_LC(paths['AG_LC'])
    
    # districts
    shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                             '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                             'IND_adm2')
    
    shpfile = Shape(region, shapefile=shapefile)
    lon_min, lat_min, lon_max, lat_max = shpfile.bbox
    lons = res_lons[np.where((res_lons>=lon_min) & (res_lons<=lon_max))]
    lats = res_lats[np.where((res_lats>=lat_min) & (res_lats<=lat_max))]
    
    start_date = datetime(2011,1,1)
    end_date = datetime(2015,8,31)
    
    results1 = []
    results2 = []
    results3 = []
    results4 = []
    for lon in lons:
        for lat in lats:
            #print lon, lat
            nearest_lon = find_nearest(lc_lons, lon)
            nearest_lat = find_nearest(lc_lats, lat)
            lon_idx = np.where(lc_lons==nearest_lon)[0]
            lat_idx = np.where(lc_lats==nearest_lat)[0]
            lc = lccs[lat_idx, lon_idx]

            if lc == 0:
                print 'lc mask used'
                continue
                
            swi_path = paths['SWI']
            vi_path = paths[vi_str]
            
            swi_list = [t_val]
            swi_df, _, _ = read_ts(swi_path, lon=lon, lat=lat, params=swi_list, 
                                   start_date=start_date, end_date=end_date)
            vi_all, _, _ = read_ts(vi_path, lon=lon, lat=lat, params=vi_str, 
                                   start_date=start_date, end_date=datetime.today())
            vi_df = vi_all[:end_date]
            vi_all = rescale_peng(vi_all, np.nanmin(vi_all), np.nanmax(vi_all))
            
            # consider leap years
            leap_years = np.where((vi_df.index.year%4 == 0) & 
                                  (vi_df.index.month >= 3))[0]
            new_idx = vi_df.index[leap_years] + timedelta(1)
            idx_array = np.array(vi_df.index)
            idx_array[leap_years] = new_idx
            vi_df.index = idx_array

            vi_df[vi_str][np.where((vi_df<0)|(vi_df>1))[0]] = np.NaN
            if len(np.where(~np.isnan(vi_df[vi_str]))[0]) == 0:
                print 'Time series is NaN'
                continue
            
            vi_min = np.nanmin(vi_df)
            vi_max = np.nanmax(vi_df)
            vi = rescale_peng(vi_df, vi_min, vi_max)
            
            swi = rescale_peng(swi_df, np.nanmin(swi_df), np.nanmax(swi_df))
            
            # calculate differences between VIs of consecutive months
            dvi = np.ediff1d(vi, to_end=np.NaN)
            vi['D_VI'] = pd.Series(dvi, index=vi.index)
            matched_data = temp_match.matching(swi, vi)
            
            if mode == 'calc_kd':
                kd = calc_kd(swi, vi, matched_data)
                path = ('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+
                        '01_kd_param_'+str(end_date.year)+
                        str(end_date.month).zfill(2)+
                        str(end_date.day).zfill(2)+'\\')
                if not os.path.exists(path):
                    os.mkdir(path)
                np.save(os.path.join(path, str(lon)+'_'+str(lat)+'.npy', kd))
            elif mode == 'pred':
                kd = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+
                             '01_kd_param\\'+str(lon)+'_'+str(lat)+'.npy').item()
                results1, results2, results3, results4 = predict_vegetation(lon, 
                                                        lat, swi, vi, 
                                                        matched_data, kd, 
                                                        vi_min, vi_max,
                                                        results1=results1,
                                                        results2=results2,
                                                        results3=results3,
                                                        results4=results4,
                                                        vi_all=vi_all)
    
    if mode == 'pred':
        ts1 = np.unique(np.array(results1)[:,3])[0]
        ts1_str = str(ts1.year).zfill(4)+str(ts1.month).zfill(2)+str(ts1.day).zfill(2)
        ts2 = np.unique(np.array(results2)[:,3])[0]
        ts2_str = str(ts2.year).zfill(4)+str(ts2.month).zfill(2)+str(ts2.day).zfill(2)
        ts3 = np.unique(np.array(results3)[:,3])[0]
        ts3_str = str(ts3.year).zfill(4)+str(ts3.month).zfill(2)+str(ts3.day).zfill(2)
        ts4 = np.unique(np.array(results4)[:,3])[0]
        ts4_str = str(ts4.year).zfill(4)+str(ts4.month).zfill(2)+str(ts4.day).zfill(2)
        np.save('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+
                '02_results\\'+str(region)+'_'+ts1_str+'.npy', 
                results1)
        np.save('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+
                '02_results\\'+str(region)+'_'+ts2_str+'.npy', 
                results2)
        np.save('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+
                '02_results\\'+str(region)+'_'+ts3_str+'.npy', 
                results3)
        np.save('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+
                '02_results\\'+str(region)+'_'+ts4_str+'.npy', 
                results4) 


def calc_kd(swi, vi, matched_data, t_val='SWI_040', vi_str='NDVI', 
             plot_fig=False):
    
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

    # calculate parameters k and d based on data until 2015
    sim_end = '2015'
    
    grouped_data = matched_data[:sim_end].groupby([matched_data[:sim_end].index.month, 
                                                   matched_data[:sim_end].index.day])
      
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
            
    return kd
            
      
def predict_vegetation(lon, lat, swi, vi, matched_data, kd, vi_min, vi_max, 
                       t_val='SWI_040', vi_str='NDVI', plot_fig=False, 
                       results1=[], results2=[], results3=[], results4=[],
                       vi_all=None):
    
    vi_sim = []
    for i in range(0,len(matched_data)):
        try:
            k, d = kd[(matched_data.index[i].month,
                       matched_data.index[i].day)]
        except KeyError:
            if len(vi_sim) > 0:
                vi_sim.append(vi_sim[i-1])
            else:
                vi_sim.append(np.NaN)
            print 'no k, d values for '+str((matched_data.index[i].month,
                                             matched_data.index[i].day))
            continue

        if len(vi_sim) == 0:
            prev_date = 8
        else:
            prev_date = (matched_data.index[i]-matched_data.index[i-1]).days
        prev_lim = 20
        if prev_date > prev_lim: # days to latest available vi value
            vi_prev = np.NaN # NaN if latest vi value is older than prev_date
            vi_sim[-1] = np.NaN # otherwise june predicts october for example
        else:
            # use vi instead of vi_sim to keep the forecast length of 10 days, assume vi of last dekade is available. what if vi_prev is nan and vi of last dekad is nan, only nans in rest of prediction
            vi_prev = matched_data[vi_str][i]
        vi_sim1 = vi_prev + k*matched_data[t_val][i] + d
        #print vi_prev, k, d, vi_sim1
        vi_sim.append(vi_sim1)
    
    # calculate SWI climatology
    clim = swi[t_val].groupby([swi[t_val].index.month, 
                               swi[t_val].index.day]).mean()
    
    # predict further into future
    new_idx = matched_data.index[1:]
    new_timestamp = matched_data.index[-1]
    for i in range(4):
        if new_timestamp.month == 12 and new_timestamp.day == 27:
            new_timestamp = datetime(new_timestamp.year+1, 1, 1)
        else:
            new_timestamp = new_timestamp + timedelta(8)
        try:
            k, d = kd[(new_timestamp.month,
                       new_timestamp.day)]
        except KeyError:
            if len(vi_sim) > 0:
                vi_sim.append(vi_sim[i-1])
            else:
                vi_sim.append(np.NaN)
            print 'no k, d values for '+str((new_timestamp.month,
                                             new_timestamp.day))
            continue
        vi_prev = vi_sim[-1]
        vi_sim1 = vi_prev + k*clim[new_timestamp.month, new_timestamp.day] + d
        vi_sim.append(vi_sim1)
        new_idx = new_idx.append(pd.Series(index=[new_timestamp]).index)
    
    # plot results
    # vi_sim is shifted 8 days compared to matched_data
    if len(new_idx) == 0:
        return results1, results2, results3, results4
    # if last date is Dec. 27th, shift it to Jan 1st instead of adding 8 days
    if new_idx[-1].day == 27 and new_idx[-1].month == 12:
            new_idx = new_idx.append(pd.Index([datetime(new_idx[-1].year+1, 
                                                        1, 1)]))
    else:
        new_idx = new_idx.append(pd.Index([new_idx[-1]+timedelta(8)]))
    
    df_sim = pd.DataFrame(vi_sim, columns=[vi_str+'_sim'], index=new_idx)
    
    plt.figure(figsize=[28,18])
    ax=vi['NDVI'].plot(color='g')
    df_sim[:-4].plot(color='k', ax=ax)
    df_sim[-5:].plot(color='r', ax=ax)
    vi_all[vi.index[-1]:].plot(color='b', ax=ax)
    lgd = plt.legend(['NDVI_orig', 'NDVI_calib', 'NDVI_sim', 'NDVI_valid'],
                     loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.title('lon: '+str(lon)+', lat: '+str(lat), fontsize=24)
    plt.grid()
    xlabels = ax.get_xticklabels()
    plt.setp(xlabels, rotation=45, fontsize=20)
    ylabels = ax.get_yticklabels()
    plt.setp(ylabels, fontsize=20)
    plt.savefig('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+
                '03_plots_1509\\'+str(lon)+'_'+str(lat)+'.png', 
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    
    # scale back and consider data gaps
    vi = vi[vi_str]
    
    res1 = vi_sim[-4]*(vi_max - vi_min)/100 + vi_min
    res2 = vi_sim[-3]*(vi_max - vi_min)/100 + vi_min
    res3 = vi_sim[-2]*(vi_max - vi_min)/100 + vi_min
    res4 = vi_sim[-1]*(vi_max - vi_min)/100 + vi_min
    results1.append([lon, lat, res1, new_idx[-4]])
    results2.append([lon, lat, res2, new_idx[-3]])
    results3.append([lon, lat, res3, new_idx[-2]])
    results4.append([lon, lat, res4, new_idx[-1]])

    return results1, results2, results3, results4


if __name__ == '__main__':
    
    # no stack data
    #ndvi_path = "E:\\poets\\RAWDATA\\NDVI_8daily_500\\"
    #swi_path = "E:\\poets\\RAWDATA\\SWI_daily_01\\"
    AG_LC = 'C:\\Users\\i.pfeil\\Desktop\\Isabella\\Peejush\\AG_Mask\\AG_LC_West_SA_0.1.nc'
    
    # stacked data
    ndvi_path = "E:\\poets\\RAWDATA\\NDVI_stack\\NDVI_gapfree.nc"
    swi_path = "E:\\poets\\RAWDATA\\SWI_daily_stack\\SWI_daily_stack.nc"
    
    paths = {'SWI': swi_path,'NDVI': ndvi_path, 'AG_LC': AG_LC}

    region = 'IN.MH.JN'
    print datetime.now()
    print 'Calculating kd...'
    start_pred(paths, region, mode='calc_kd')
    print 'Prediction...'
    start_pred(paths, region, mode='pred') 
    
    print 'Plot results...'
    pred1 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\IN.MH.JN_20150805.npy')
    pred2 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\IN.MH.JN_20150813.npy')
    pred3 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\IN.MH.JN_20150821.npy')
    pred4 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\IN.MH.JN_20150901.npy')   
    validate_prediction(pred1, ndvi_path, plotname=region+'_Prediction')
    validate_prediction(pred2, ndvi_path, plotname=region+'_Prediction')
    validate_prediction(pred3, ndvi_path, plotname=region+'_Prediction')
    validate_prediction(pred4, ndvi_path, plotname=region+'_Prediction')
    
    print 'done', datetime.now()