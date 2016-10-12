import os
import numpy as np
import pandas as pd
from scipy import signal as sg
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pytesmo.temporal_matching as temp_match
from poets.shape.shapes import Shape

from Basemap_scatterplot import scatter_subplots
from data_analysis import rescale_peng
from readers import read_ts, read_img, find_nearest, read_AG_LC
from simon import read_ts_area


def validate_prediction(pred, vi_path, plotname):

    lons = np.around(pred[:,0].astype(np.double),2)
    lats = np.around(pred[:,1].astype(np.double),2)
    data = pred[:,2]
    timestamp = np.unique(pred[:,3])[0]
    
    # districts
    shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                             '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                             'IND_adm2')
    
    shpfile = Shape(region, shapefile=shapefile)
    lon_min, lat_min, lon_max, lat_max = shpfile.bbox
    
    vi_ts = datetime(timestamp.year, timestamp.month, timestamp.day)
    vi_data, vi_lons, vi_lats, _ = read_img(vi_path, param='NDVI', 
                                                  lat_min=lat_min, lat_max=lat_max,
                                                  lon_min=lon_min, lon_max=lon_max, 
                                                  timestamp=vi_ts)

    vi_data = vi_data[0,:,:]
    vi_data = np.ma.masked_where(vi_data<0, vi_data)

    vi_lon, vi_lat = np.meshgrid(vi_lons, vi_lats)
    vi_lon = vi_lon.flatten()
    vi_lat = vi_lat.flatten()
    
    #print data.min(), data.max(), data.mean(), data.std()
    #print vi_data.min(), vi_data.max(), vi_data.mean(), vi_data.std()
    
    # reshape data to 2D array
    latsize = len(np.unique(lats))
    lonsize = len(np.unique(lons))
    data_reshape = np.zeros((latsize, lonsize)) - 99
    lat_mesh, lon_mesh = np.meshgrid(np.unique(lats)[::-1], np.unique(lons))
    for i in range(len(lons)):
        lon_idx, lat_idx = np.where((lat_mesh == lats[i])&(lon_mesh == lons[i]))
        if len(lat_idx) == 0 or len(lon_idx) == 0:
            continue
        data_reshape[lat_idx, lon_idx] = data[i]
    data_pred = np.ma.masked_where(data_reshape==-99, data_reshape)
     
    #corr(data_pred, vi_data, vi_date)
    
    #===========================================================================
    # scatter_subplots(lons, lats, data, 350, 
    #                  vi_lon, vi_lat, vi_data, 1, plotname,
    #                  llcrnrlat=lat_min-0.5, urcrnrlat=lat_max+0.5,
    #                  llcrnrlon=lon_min-0.5, urcrnrlon=lon_max+0.5,
    #                  vi_date=vi_date)
    #===========================================================================
     
    return pred


def pred_mean_ts(pred1, pred2, pred3, pred4, plotname):
    
    data1 = pred1[:,2].mean()
    timestamp1 = np.unique(pred1[:,3])[0]
    data2 = pred2[:,2].mean()
    timestamp2 = np.unique(pred2[:,3])[0]
    data3 = pred3[:,2].mean()
    timestamp3 = np.unique(pred3[:,3])[0]
    data4 = pred4[:,2].mean()
    timestamp4 = np.unique(pred4[:,3])[0]
    
    data = np.array([data1, data2, data3, data4])
    timestamp = np.array([timestamp1, timestamp2, timestamp3, timestamp4])
    
    pred_df = pd.DataFrame(data, columns=['pred_mean'], index=[timestamp])
    
    # districts
    shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                             '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                             'IND_adm2')
    region = plotname[:8]
    print region
    shpfile = Shape(region, shapefile=shapefile)
    lon_min, lat_min, lon_max, lat_max = shpfile.bbox
    
    # ndvi gapfree 500m
    vi_path = "E:\\poets\\RAWDATA\\NDVI_stack\\NDVI_gapfree.nc"
    vi_all = read_ts_area(vi_path, param='NDVI', 
                         lat_min=lat_min, 
                         lat_max=lat_max, 
                         lon_min=lon_min, 
                         lon_max=lon_max)
    
    vi_all = vi_all['2013':'2015']
    idx = np.where(vi_all['NDVI'] != 0)[0]
    vi_data = vi_all.values[idx]
    vi_index = vi_all.index[idx]
    
    gapfree_df = pd.DataFrame(data=vi_data, columns=['NDVI_mean_gapfree'], index=[vi_index])
    
    ax = gapfree_df.plot()
    pred_df.plot(ax=ax)
    plt.title(plotname, fontsize=22)
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.show()
    plt.savefig('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\06_analysis\\'+plotname+
                '.png', bbox_inches='tight')
    plt.close()
    
    print 'done'


def corr(i01, i02, vi_date):
    # from http://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images
    
    i1 = i01.data
    i1[np.where(i01.mask == True)] = np.NAN
    i2 = i02.data
    i2[np.where(i02.mask == True)] = np.NAN
    
    #===========================================================================
    # print i1
    # print np.nansum(i1), np.nanmean(i1), np.nanstd(i1)
    # print i2
    # print np.nansum(i2), np.nanmean(i2), np.nanstd(i2)
    #===========================================================================
    
    notnan_size = i1.size-np.where(np.isnan(i1-i2))[0].size
    
    dist_euclidean = np.sqrt(np.nansum((i1 - i2)**2))/notnan_size

    dist_manhattan = np.nansum(abs(i1 - i2)) / notnan_size

    dist_ncc = (np.nansum((i1 - np.nanmean(i1)) * (i2 - np.nanmean(i2))) / 
                ((notnan_size - 1) * np.nanstd(i1) * np.nanstd(i2)))
    
    absdiff = abs(i1 - i2)
    
    print 'dist_euclidean, dist_manhattan, dist_ncc:'
    print dist_euclidean, dist_manhattan, dist_ncc
    
    #===========================================================================
    # plt.matshow(absdiff)
    # plt.colorbar()
    # plt.show()
    # #plt.savefig('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\05_stats\\'+
    # #            'absdiff_'+str(vi_date.year)+str(vi_date.month).zfill(2)+
    # #            str(vi_date.day).zfill(2)+'.png', bbox_inches='tight')
    #===========================================================================


def start_pred(paths, region, end_date, vi_str='NDVI', t_val='SWI_040',
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
    
    start_date = datetime(2007,1,1)
    
    results1 = []
    results2 = []
    results3 = []
    results4 = []
    for lon in lons:
        for lat in lats:
            print lon, lat
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
            vi_all = read_ts_area(vi_path, vi_str, lat_min=nearest_lat-0.04, 
                                  lat_max=nearest_lat+0.04, 
                                  lon_min=nearest_lon-0.04, 
                                  lon_max=nearest_lon+0.04)
            #vi_all, _, _ = read_ts(vi_path, lon=lon, lat=lat, params=vi_str, 
            #                       start_date=start_date, end_date=datetime.today())
            vi_df = vi_all[start_date:end_date]
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
            
            if mode == 'calc_kd':
                matched_data = temp_match.matching(swi, vi)
                kd = calc_kd(swi, vi, matched_data)
                path = ('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+
                        '01_kd_param_'+str(end_date.year)+
                        str(end_date.month).zfill(2)+
                        str(end_date.day).zfill(2)+'_'+region[-2:]+'\\')
                if not os.path.exists(path):
                    os.mkdir(path)
                np.save(os.path.join(path, str(lon)+'_'+str(lat)+'.npy'), kd)
            elif mode == 'pred':
                try:
                    kd = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+
                             '01_kd_param_'+str(end_date.year)+
                        str(end_date.month).zfill(2)+
                        str(end_date.day).zfill(2)+'_'+region[-2:]+'\\'+str(lon)+'_'+str(lat)+'.npy').item()
                except IOError:
                    print 'No kd file'
                    continue
                matched_data = temp_match.matching(swi, vi[vi_str])
                results1, results2, results3, results4 = predict_vegetation(lon, 
                                                        lat, swi, vi, 
                                                        matched_data, kd, 
                                                        vi_min, vi_max,
                                                        results1=results1,
                                                        results2=results2,
                                                        results3=results3,
                                                        results4=results4,
                                                        vi_all=vi_all,
                                                        region=region,
                                                        end_date=end_date)
    
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
    sim_end = '2014'
    
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
                       vi_all=None, region='IN.MH.JN', 
                       end_date=datetime(2015,8,31)):
    
    vi_sim = []
    for i in range(0,len(matched_data[1:])):
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
    clim_reset = clim.reset_index()
    idx_8d = []
    for doy in range(len(clim_reset)):
        month = clim_reset['level_0'].values[doy]
        day = clim_reset['level_1'].values[doy]
        if month == 2 and day == 29:
            continue
        idx_8d.append(datetime(2010,month,day))
    clim_365 = clim.iloc[[count for count, _ in enumerate(clim) if count != 59]]
    clim_swi_tmp = pd.DataFrame(clim_365.values, index=idx_8d, columns=[t_val],
                                dtype=clim.dtype)
    clim_swi_tmp = clim_swi_tmp.resample('8D').mean()
    clim_swi = clim_swi_tmp[t_val].groupby([clim_swi_tmp[t_val].index.month, 
                                            clim_swi_tmp[t_val].index.day]).mean()
    
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
        if i == 0: # letzten NDVI wert nehmen, nicht letzten vi_sim!
            vi_prev = vi[vi_str][-1]
        else:
            vi_prev = vi_sim[-1]
        vi_sim1 = vi_prev + k*clim_swi[new_timestamp.month, 
                                       new_timestamp.day] + d
        vi_sim.append(vi_sim1)
        new_idx = new_idx.append(pd.Series(index=[new_timestamp]).index)
    
    # plot results
    # vi_sim is shifted 8 days compared to matched_data
    if len(new_idx) == 0:
        return results1, results2, results3, results4    
    
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
    
    plotname = str(end_date.year-2000)+str(end_date.month).zfill(2)+'_'+region[-2:]
    save_path = 'C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\03_plots_'+plotname
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, str(lon)+'_'+str(lat)+'.png'), 
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
    
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

    regions = ['IN.MH.JN']
    end_dates = [datetime(2013,5,31), datetime(2015,7,31), datetime(2015,8,31)]
    
    
    #print 'Calculating kd...', datetime.now()
    #start_pred(paths, region, mode='calc_kd') 
    
    #print 'Plot results...', datetime.now()
    # plot: set vmin vmax accordingly
    
    for region in regions:
        #=======================================================================
        # pred1 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20130602.npy')
        # pred2 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20130610.npy')
        # pred3 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20130618.npy')
        # pred4 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20130626.npy')
        # #pred_mean_ts(pred1, pred2, pred3, pred4, plotname=plotname)
        #=======================================================================
         
        pred5 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20150805.npy')
        pred6 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20150813.npy')
        pred7 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20150821.npy')
        pred8 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20150829.npy')
        #pred_mean_ts(pred5, pred6, pred7, pred8, plotname=region+'_201508')
         
        pred9 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20150906.npy')
        pred10 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20150914.npy')
        pred11 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20150922.npy')
        pred12 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\'+region+'_20150930.npy')
        #pred_mean_ts(pred9, pred10, pred11, pred12, plotname=region+'_201509')
         
        #=======================================================================
        # plotname = region+'_201306'
        # print plotname
        # validate_prediction(pred1, ndvi_path, plotname=plotname+'02')
        # validate_prediction(pred2, ndvi_path, plotname=plotname+'10')
        # validate_prediction(pred3, ndvi_path, plotname=plotname+'18')
        # validate_prediction(pred4, ndvi_path, plotname=plotname+'26')
        #=======================================================================
         
        plotname = region+'_201508'
        print plotname
        validate_prediction(pred5, ndvi_path, plotname=plotname+'05')
        validate_prediction(pred6, ndvi_path, plotname=plotname+'13')
        validate_prediction(pred7, ndvi_path, plotname=plotname+'21')
        validate_prediction(pred8, ndvi_path, plotname=plotname+'29')
         
        plotname = region+'_201509'
        print plotname
        validate_prediction(pred9, ndvi_path, plotname=plotname+'06')
        validate_prediction(pred10, ndvi_path, plotname=plotname+'14')
        validate_prediction(pred11, ndvi_path, plotname=plotname+'22')
        validate_prediction(pred12, ndvi_path, plotname=plotname+'30')
    
    
    print 'done', datetime.now()