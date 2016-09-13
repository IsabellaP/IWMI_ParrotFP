import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pytesmo.temporal_matching as temp_match
from poets.shape.shapes import Shape

from readers import read_ts, read_img
from data_analysis import rescale_peng
from Basemap_scatterplot import scatterplot


def validate_prediction():
    
    pred = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\results.npy')
    lons = pred[:,0]
    lats = pred[:,1]
    data = pred[:,2]
    timestamp = np.unique(pred[:,3])[0]
    data_resc = rescale_peng(data, min(data), max(data))
    scatterplot(lons, lats, data_resc, s=200, title=str(timestamp)+' - simulated', 
                discrete=False, vmin=0, vmax=100)
    
    # districts
    shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                             '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                             'IND_adm2')
    
    shpfile = Shape(region, shapefile=shapefile)
    lon_min, lat_min, lon_max, lat_max = shpfile.bbox
    
    vi_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA2\\NDVI01\\West_SA_0.1_dekad.nc"
    vi_ts = datetime(timestamp.year, timestamp.month, timestamp.day)
    vi_data, vi_lons, vi_lats, vi_date = read_img(vi_path, param='NDVI_dataset', 
                                                  lat_min=lat_min, lat_max=lat_max,
                                                  lon_min=lon_min, lon_max=lon_max, 
                                                  timestamp=vi_ts)
    
    vi_resc = rescale_peng(vi_data.flatten(), np.nanmin(vi_data.flatten()), 
                           np.nanmax(vi_data.flatten()))
    vi_lon, vi_lat = np.meshgrid(vi_lons, vi_lats)
    vi_lon = vi_lon.flatten()
    vi_lat = vi_lat.flatten()
    
    scatterplot(vi_lon, vi_lat, vi_resc, s=200, title=str(vi_date)+' - orig. data', 
                discrete=False, vmin=0, vmax=100)
    
    return pred

def start_district_pred(paths, region, pred_date, vi_str='NDVI_dataset', 
                        t_val='SWI_040', monthly=False):
    
    with Dataset(paths['SWI'], 'r') as ncfile:
        swi_lons = ncfile.variables['lon'][:]
        swi_lats = ncfile.variables['lat'][:]
    
    # districts
    shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                             '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                             'IND_adm2')
    
    shpfile = Shape(region, shapefile=shapefile)
    lon_min, lat_min, lon_max, lat_max = shpfile.bbox
    lons = swi_lons[np.where((swi_lons>=lon_min) & (swi_lons<=lon_max))]
    lats = swi_lats[np.where((swi_lats>=lat_min) & (swi_lats<=lat_max))]
    
    start_date = datetime(2007,7,1)
    end_date = datetime(2015,7,1)
    
    results = []
    for lon in lons:
        for lat in lats:
            print lon, lat
                
            swi_path = paths['SWI']
            vi_path = paths[vi_str]
            
            swi_list = [t_val]
            swi_df = read_ts(swi_path, lon=lon, lat=lat, params=swi_list, 
                             start_date=start_date, end_date=end_date)
            vi_all = read_ts(vi_path, lon=lon, lat=lat, params=vi_str, 
                         start_date=start_date, end_date=end_date)
            vi_all[vi_str][np.where(vi_all==-99)[0]] = np.NaN
            
            vi = vi_all[:pred_date]
            vi = rescale_peng(vi, np.nanmin(vi), np.nanmax(vi))
            
            swi_all = swi_df[t_val]
            swi = swi_all[:pred_date]
            swi = rescale_peng(swi, np.nanmin(swi), np.nanmax(swi))
    
            # resample monthly
            if monthly:
                swi = swi.resample("M").mean()
                vi = vi.resample("M").mean()
            
            # calculate differences between VIs of consecutive months
            dvi = np.ediff1d(vi, to_end=np.NaN)
            vi['D_VI'] = pd.Series(dvi, index=vi.index)
            matched_data = temp_match.matching(swi, vi)
            
            kd = zribi_kd(swi, vi, matched_data)
            results = zribi_sim(lon, lat, swi, vi, matched_data, kd, 
                                results=results)
        
    np.save('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\results.npy', 
             results)


def zribi_kd(swi, vi, matched_data, t_val='SWI_040', vi_str='NDVI', 
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

    # calculate parameters k and d only from 2007-2012 data
    sim_start = '2007'
    sim_end = '2012'
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
            
    return kd
            
      
def zribi_sim(lon, lat, swi, vi, matched_data, kd, t_val='SWI_040', 
              vi_str='NDVI_dataset', plot_fig=False, monthly=False, results=[]):
    
    vi_sim = []
    for i in range(0,len(matched_data)):
        #print i, matched_data.index[i]
        try:
            if monthly:
                k, d = kd[matched_data.index[i].month]
            else:
                k, d = kd[(matched_data.index[i].month,
                           matched_data.index[i].day)]
        except KeyError:
            if len(vi_sim) > 0:
                vi_sim.append(vi_sim[i-1])
            else:
                vi_sim.append(np.NaN)
            if monthly:
                print 'no k, d values for month '+str(matched_data.index[i].month)
            else:
                print 'no k, d values for '+str((matched_data.index[i].month,
                                                 matched_data.index[i].day))
            continue
        
        if len(vi_sim) == 0:
            prev_date = 10
        else:
            prev_date = (matched_data.index[i]-matched_data.index[i-1]).days
        if monthly:
            prev_lim = 60
        else:
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
    
    # plot results
    if monthly:
        results = pd.DataFrame(vi[vi_str].values*100, columns=[vi_str], index=vi.index)
        results[t_val] = pd.Series(swi.values, index=swi.index)
        results[vi_str+'_sim'] = pd.Series(np.multiply(vi_sim, 100), 
                                        index=matched_data.index)
        
    else:
        # vi_sim is shifted 1 dekade comapred to matched_data
        new_idx = matched_data.index[1:]
        if new_idx[-1].day == 21:
            if new_idx[-1].month == 12:
                new_idx = new_idx.append(pd.Index([datetime(new_idx[-1].year+1, 
                                                            1, 1)]))
            else:
                new_idx = new_idx.append(pd.Index([datetime(new_idx[-1].year, 
                                                  new_idx[-1].month+1, 1)]))
        else:
            new_idx = new_idx.append(pd.Index([new_idx[-1]+timedelta(10)]))
        
        df_sim = pd.DataFrame(vi_sim, columns=[vi_str+'_sim'], index=new_idx)
    
    results.append([lon, lat, vi_sim[-1], new_idx[-1]])
    
    #===========================================================================
    # ax=matched_data[t_val].plot()
    # matched_data[vi_str].plot(ax=ax)
    # df_sim.plot(ax=ax)
    # plt.title('Lon: '+str(lon)+', lat: '+str(lat)+', t value: '+t_val)
    # plt.legend([t_val, vi_str, 'vi_sim'], loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    # #plt.savefig('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\'+str(lon)+
    # #            '_'+str(lat)+'.png', bbox_inches='tight')
    #  
    # print df_sim
    #===========================================================================
    return results


if __name__ == '__main__':
    
    # set paths to datasets
    ssm_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ssm\\foxy_finn\\R1A\\"
    lcpath = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA2\\LC\\ESACCI-LC-L4-LCCS-Map-300m-P5Y-20100101-West_SA-v1.6.1.nc"
    ndvi_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA2\\NDVI\\NDVI_stack.nc"
    ndvi01_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA2\\NDVI01\\West_SA_0.1_dekad.nc"
    ndvi300_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\NDVI300\\"
    lai_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA2\\LAI\\LAI_stack.nc"
    swi_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA2\\SWI\\SWI_stack.nc"
    fapar_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA2\\FAPAR\\FAPAR_stack.nc"
    
    paths = {'ssm': ssm_path, 'lc': lcpath, 'NDVI300': ndvi300_path, 
             'NDVI_dataset': ndvi01_path, 'LAI': lai_path, 'SWI': swi_path, 
             'FAPAR': fapar_path}
    
    region = 'IN.MH.JN'
    pred_date = datetime(2012,4,1)
    #start_district_pred(paths, region, pred_date)    
    validate_prediction()
