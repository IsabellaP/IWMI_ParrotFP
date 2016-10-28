import os
import numpy as np
import pandas as pd
import ast
import calendar
from datetime import datetime, timedelta
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import pytesmo.temporal_matching as temp_match
from poets.shape.shapes import Shape

from veg_pred_readers import read_ts_area
from nc_stack_uptodate import array_to_raster
from veg_pred_preprocessing import read_cfg, unzip
from nc_stack_uptodate import check_stack, check_tiff_stack


def start_pred(paths, region, cfg, results_path,
               end_date=datetime.today(), vi_str='NDVI', t_val='SWI_040',
               last_year=2015):
    """
    Start parameter calculation and vegetation prediction. If mode is set to
    'calc_kd', parameters for vegetation prediction are calculated and saved.
    If mode is set to 'pred', vegetation prediction is started for given date
    argument (end_date).
    
    Parameters:
    -----------
    paths : dict
        Paths to VI and SWI datasets (datasets as stacks)
    region : str
        Region as str, as in provided shapefile, e.g. 'IN.MH.JN' (Jalna)
    cfg : dict
        Settings from cfg-file
    results_path : str
        Path where all results are saved
    end_date : datetime, optional
        Date until data is available, e.g. 1 Oct 2016: datetime(2016,10,1), 
        default: today
    vi_str : str, optional
        Variable name of VI dataset in nc-stack, default: 'NDVI'
    t_val : str, optional
        SWI t-value, default: 'SWI_040'
    last_year : int
        Last full year for which data is available
    """
    
    # get bounding box for district
    shp_path = cfg['shp_path']
    if shp_path == 'None':
        latlon = cfg['latlon'].split()
        lat_min = float(latlon[0])
        lat_max = float(latlon[1])
        lon_min = float(latlon[2])
        lon_max = float(latlon[3])
    else:
        shpfile = Shape(region, shapefile=shp_path)
        lon_min, lat_min, lon_max, lat_max = shpfile.bbox
        
    start_date = datetime(2007,1,1)
    
    # get lons and lats
    with Dataset(paths['SWI'], 'r') as ncfile:
        res_lons = ncfile.variables['lon'][:]
        res_lats = ncfile.variables['lat'][:]
        lons_idx = np.where((res_lons>=lon_min) & (res_lons<=lon_max))[0]
        lats_idx = np.where((res_lats>=lat_min) & (res_lats<=lat_max))[0]
        lons = res_lons[lons_idx]
        lats = res_lats[lats_idx]
        
        # date period
        unit_temps = ncfile.variables['time'].units
        nctime = ncfile.variables['time'][:]
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError:  # Attribute doesn't exist
            cal_temps = u"gregorian"  # or standard
        
        all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)
        date_idx = np.where((all_dates >= start_date) &
                            (all_dates <= end_date))[0]

        swi_data = ncfile.variables[t_val][date_idx, lats_idx, lons_idx]
     
    results1 = []
    results2 = []
    results3 = []
    results4 = []
    i = 0
    for lon in lons:
        for lat in lats:
            i += 1
            if i % 5 == 0:
                print str(float(i)/(len(lats)*len(lons)) * 100) + '%'
                
            vi_path = paths[vi_str]
             
            # read SWI data
            lon_idx = np.where(lons == lon)[0]
            lat_idx = np.where(lats == lat)[0]
            swi_single = swi_data[:, lat_idx, lon_idx]
            swi_df = pd.DataFrame(swi_single, index=all_dates[date_idx],
                                  columns=[t_val])
            
            swi_df.values[np.where(swi_df.values == 255)] = np.NAN
             
            # read VI-data for 0.1 degree area
            vi_all = read_ts_area(vi_path, vi_str, 
                                  lat_min=lat-0.05, lat_max=lat+0.05, 
                                  lon_min=lon-0.05, lon_max=lon+0.05)
            vi_df = vi_all[start_date:end_date]
             
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
             
            # apply minmax-scaling to datasets
            vi_min = np.nanmin(vi_df)
            vi_max = np.nanmax(vi_df)
            vi = rescale_minmax(vi_df, vi_min, vi_max)
            swi = rescale_minmax(swi_df, np.nanmin(swi_df), np.nanmax(swi_df))
             
            # calculate differences between VIs of consecutive months
            dvi = np.ediff1d(vi, to_end=np.NaN)
            vi['D_VI'] = pd.Series(dvi, index=vi.index)
             
            # calculate and save kd-parameters
            sim_end = str(last_year).zfill(4)
            kd_path = (results_path+ '01_kd_param_'+sim_end+'_'+region+'\\')
            if not os.path.exists(kd_path):
                os.mkdir(kd_path)
            if not os.path.exists(kd_path+'\\'+str(lon)+'_'+str(lat)+'.npy'):
                #print 'Calculate kd...'
                matched_data = temp_match.matching(swi, vi)
                kd = calc_kd(swi, vi, matched_data, sim_end=end_date)
                np.save(os.path.join(kd_path, str(lon)+'_'+str(lat)+'.npy'), kd)
            else:
                pass
                #print 'kd already exist.'
             
            # predict VI values
            try:
                kd = np.load(results_path+ '01_kd_param_'+sim_end+'_'+
                             region+'\\'+str(lon)+'_'+str(lat)+'.npy').item()
            except IOError:
                #print 'No kd file'
                continue
            #print 'Predict...'
            matched_data = temp_match.matching(swi, vi[vi_str])
            results1, results2, results3, results4 = predict_vegetation(lon, 
                                                    lat, swi, vi, kd, 
                                                    vi_min, vi_max,
                                                    region=region,
                                                    end_date=end_date,
                                                    results_path=results_path,
                                                    results1=results1,
                                                    results2=results2,
                                                    results3=results3,
                                                    results4=results4)
     
    # save results of prediction
    if not os.path.exists(results_path+'02_results\\'):
        os.mkdir(results_path+'02_results\\')
    ts1 = np.unique(np.array(results1)[:,3])[0]
    ts1_str = str(ts1.year).zfill(4)+str(ts1.month).zfill(2)+str(ts1.day).zfill(2)
    ts2 = np.unique(np.array(results2)[:,3])[0]
    ts2_str = str(ts2.year).zfill(4)+str(ts2.month).zfill(2)+str(ts2.day).zfill(2)
    ts3 = np.unique(np.array(results3)[:,3])[0]
    ts3_str = str(ts3.year).zfill(4)+str(ts3.month).zfill(2)+str(ts3.day).zfill(2)
    ts4 = np.unique(np.array(results4)[:,3])[0]
    ts4_str = str(ts4.year).zfill(4)+str(ts4.month).zfill(2)+str(ts4.day).zfill(2)
    np.save(results_path + '02_results\\'+str(region)+'_'+ts1_str+'.npy', 
            results1)
    np.save(results_path + '02_results\\'+str(region)+'_'+ts2_str+'.npy', 
            results2)
    np.save(results_path + '02_results\\'+str(region)+'_'+ts3_str+'.npy', 
            results3)
    np.save(results_path + '02_results\\'+str(region)+'_'+ts4_str+'.npy', 
            results4)
    
    # create average prediction time series for region
    plotname = (region+'_'+str(end_date.year)+str(end_date.month).zfill(2)+
                str(end_date.day).zfill(2))
    mean_ts(np.array(results1), np.array(results2), np.array(results3), 
            np.array(results4), lon_min, lat_min, lon_max, 
            lat_max, vi_path, vi_str, results_path, plotname)
    

def mean_ts(pred1, pred2, pred3, pred4, lon_min, lat_min, lon_max, lat_max,
            vi_path, vi_str, results_path, plotname):
    """
    Saves mean time series for observed and predicted VI over region.
    
    Parameters:
    ----------
    Pred1-4 : np.array
        Results of the prediction
    latlon_minmax : float
        Bounding box of the region
    vi_path : str
        Path to vi dataset
    vi_str : str
        Name of dataset, e.g. 'NDVI'
    results_path : str
        Path where all results are saved
    plotname : str
        Name of resulting plot
    """
    
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
    
    # read vi
    vi_all = read_ts_area(vi_path, param=vi_str, 
                         lat_min=lat_min, 
                         lat_max=lat_max, 
                         lon_min=lon_min, 
                         lon_max=lon_max)
    
    vi_all = vi_all['2013':timestamp1]
    idx = np.where(vi_all['NDVI'] != 0)[0]
    vi_data = vi_all.values[idx]
    vi_index = vi_all.index[idx]
    
    vi_df = pd.DataFrame(data=vi_data, columns=['NDVI_mean_gapfree'],
                              index=[vi_index])
    
    ax = vi_df.plot()
    pred_df.plot(ax=ax)
    plt.title(plotname, fontsize=22)
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plotpath = os.path.join(results_path, '031_mean')
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)
    plt.savefig(os.path.join(plotpath, plotname+'.png'), bbox_inches='tight')
    plt.close()


def calc_kd(swi, vi, matched_data, t_val='SWI_040', vi_str='NDVI', 
            sim_end='2015', plot_fig=False):
    
    """ Simulate VI from SWI as in Zribi et al., 2010.
    
    Parameters:
    -----------
    swi : pd.DataFrame
        SWI dataset
    vi : pd.DataFrame
        VI dataset
    matched_data : pd.DataFrame
        SWI, VI and dVI dataset
    t_val : str, optional
        T value, default: 20
    vi_str : str, optional
        Vegetation index to use, default: NDVI
    sim_end : str
        Last full year for which data is available
    plot_fig : bool, optional
        If true, functions from which k and d are derived are plotted, 
        default: False
    
    Returns:
    --------
    kd : dict
        Calculated kd-values for each (month,day)-group
    """
    
    # group data by month and day
    grouped_data = matched_data[:sim_end].groupby([matched_data[:sim_end].index.month, 
                                                   matched_data[:sim_end].index.day])
    
    # calculate parameters of linear regression between SWI and dVI
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
            
      
def predict_vegetation(lon, lat, swi, vi, kd, vi_min, vi_max, 
                       region, end_date, results_path,
                       t_val='SWI_040', vi_str='NDVI', 
                       results1=[], results2=[], results3=[], results4=[]):
    """
    lon, lat : float
        Longitude and latitude of pixel
    swi, vi: pd.DataFrame
        SWI and VI dataset
    kd : dict
        parameters of linear regression
    vi_min, vi_max : float
        Min and max of vi-timeseries before rescaling
    region : str
        Region as str, as in provided shapefile, e.g. 'IN.MH.JN' (Jalna)
    end_date : datetime
        Date until data is available
    results_path : str
        Path where all results are saved
    t_val : str, optional
        T value, default: 20
    vi_str : str, optional
        Vegetation index to use, default: NDVI
    results1-4 : list
        List where results for every lonlat-position are appended
        
    Returns:
    --------
    results1-4 : list
        List where results for every lonlat-position are appended
    """
    
    vi_sim = []    
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
    new_idx = pd.DataFrame([]).index
    new_timestamp = vi.index[-1]
    
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
        if i == 0:
            vi_prev = vi[vi_str][-1]
        else:
            vi_prev = vi_sim[-1]
        vi_sim1 = vi_prev + k*clim_swi[new_timestamp.month, 
                                       new_timestamp.day] + d
        vi_sim.append(vi_sim1)
        new_idx = new_idx.append(pd.Series(index=[new_timestamp]).index)
    
    if len(new_idx) == 0:
        return results1, results2, results3, results4
    
    df_sim = pd.DataFrame(vi_sim, columns=[vi_str+'_sim'], index=new_idx)
    
    plt.figure(figsize=[28,18])
    ax=vi['NDVI'].plot(color='g')
    df_sim.plot(color='r', ax=ax)
    lgd = plt.legend(['NDVI_orig', 'NDVI_sim'],
                     loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.title('lon: '+str(lon)+', lat: '+str(lat), fontsize=24)
    plt.grid()
    xlabels = ax.get_xticklabels()
    plt.setp(xlabels, rotation=45, fontsize=20)
    ylabels = ax.get_yticklabels()
    plt.setp(ylabels, fontsize=20)
    
    plotname = (str(end_date.year-2000)+str(end_date.month).zfill(2)+
                str(end_date.day).zfill(2)+'_'+region)
    save_path = os.path.join(results_path, '03_plots_'+plotname)
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


def save_results(pred, plotname, results_path):
    """
    Saves results as GeoTiffs.
    
    Parameters:
    ----------
    pred : np.array
        Prediction results
    plotname : str
        Name of GeoTiff
    results_path : str
        Path where all results are stored
    """
    
    lons = np.around(pred[:,0].astype(np.double),2)
    lats = np.around(pred[:,1].astype(np.double),2)
    data = pred[:,2]
    
    
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
    
    array_to_raster(data_pred, lon_mesh[:,0], lat_mesh[0], 
                    os.path.join(savepath,plotname+'.tif'))


def rescale_minmax(data, data_min, data_max):
    """
    Performs minmax-rescaling on a timeseries.
    
    Parameters:
    ----------
    data : pd.DataFrame
        Dataset
    data_min, data_max : float
        Minimum and Maximum of the dataset
        
    Returns:
    -------
    data_resc : pd.DataFrame
        rescaled dataset
    """
    
    data_resc = (data - data_min) / (data_max - data_min) * 100
    return data_resc


def add_months(sourcedate, months):
    """
    Adds a specified number of months to a given date.
    
    Parameters:
    -----------
    sourcedate : datetime
        Date
    months : int
        Number of months that should be added
        
    Returns:
    --------
    Date after months addition
    """
    
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12 )
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year,month)[1])
    return datetime(year,month,day)


if __name__ == '__main__':
    
    print 'Process started: '+str(datetime.now())
    
    # get settings from cfg-file
    cfg = read_cfg('config_file_daily.cfg')
    
    # check nc-stack availability, unzip files if necessary
    swi_zippath = cfg['swi_zippath']
    data_path = cfg['swi_rawdata']
    unzip(swi_zippath, data_path)
    
    data_path = cfg['swi_rawdata']
    data_path_nc = cfg['swi_path_nc']
    nc_stack_path = cfg['swi_path']
    swi_stack_name = cfg['swi_stack_name']
    variables = cfg['swi_variables'].split()
    datestr = ast.literal_eval(cfg['swi_datestr'])
    
    check_stack(data_path, data_path_nc, nc_stack_path, swi_stack_name, 
                variables, datestr)
    
    # check and update VI stack
    data_path = cfg['vi_rawdata']
    data_path_nc = cfg['vi_path_nc']
    nc_stack_path = cfg['vi_path']
    swi_stack_name = cfg['vi_stack_name']
    variables = cfg['vi_variables']
    datestr = ast.literal_eval(cfg['vi_datestr'])
    
    check_tiff_stack(data_path, data_path_nc, nc_stack_path, swi_stack_name, 
                     variables, datestr)
    
    # start vegetation prediction
    swi_path = cfg['swi_path'] + cfg['swi_stack_name']
    vi_path = cfg['vi_path'] + cfg['vi_stack_name']
    results_path = cfg['results_path']
    region = cfg['region']
    str_date = cfg['end_date']
    end_date = datetime.strptime(str_date, '%Y-%m-%d')
    
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    paths = {'SWI': swi_path,'NDVI': vi_path}

    # start prediction
    check_path = os.path.join(results_path, '03_plots_'+str(end_date.year-2000)+
                              str(end_date.month).zfill(2)+
                              str(end_date.day).zfill(2)+'_'+region)
    print 'Writing results to '+results_path
    print 'Delete folder 03_plots_'+region+' if not fully processed!'
    if not os.path.exists(check_path):
        print 'Start prediction for '+region+', '+str(end_date)
        start_pred(paths, region, cfg, results_path, end_date=end_date,
                   last_year=(end_date+timedelta(8)).year-1)
    
    # save results
    savepath = os.path.join(results_path, '04_geotiffs')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for result in os.listdir(os.path.join(results_path,'02_results')):
        plotname, ext = os.path.splitext(result)
        if not os.path.exists(os.path.join(savepath, plotname+'.tif')):
            print 'save '+plotname
            pred = np.load(os.path.join(results_path,'02_results',result))
            save_results(pred, plotname, savepath)
            
    print 'Process finished: '+str(datetime.now())
    