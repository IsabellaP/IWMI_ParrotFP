import numpy as np
import pandas as pd
from readers import read_ts
import pytesmo.temporal_matching as temp_match
import matplotlib.pyplot as plt


def zribi(paths, lon, lat, start_date, end_date, t_val='SWI_020', vi_str='NDVI', 
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
    
    swi_list = ['SWI_001', 'SWI_010', 'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']
    swi_df = read_ts(swi_path, lon=lon, lat=lat, params=swi_list, 
                     start_date=start_date, end_date=end_date)
    vi = read_ts(vi_path, lon=lon, lat=lat, params=vi_str, 
                 start_date=start_date, end_date=end_date)
    vi[vi_str][np.where(vi==255)[0]] = np.NaN
    
    swi = swi_df[t_val]
    
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
    if monthly:
        results = pd.DataFrame(vi[vi_str].values*100, columns=[vi_str], index=vi.index)
        results[t_val] = pd.Series(swi.values, index=swi.index)
        results[vi_str+'_sim'] = pd.Series(np.multiply(vi_sim, 100), 
                                        index=matched_data.index)
        
    else:
        results = pd.DataFrame(matched_data[t_val].values, columns=[t_val],
                               index=matched_data.index)
        results[vi_str] = pd.Series(matched_data[vi_str].values*100, 
                                    index=matched_data.index)
        results[vi_str+'_sim'] = pd.Series(np.multiply(vi_sim, 100), 
                                           index=matched_data.index)
        
    print results
    results.plot()
    plt.title('Lon: '+str(lon)+', lat: '+str(lat)+', t value: '+t_val)
    plt.show()
      
    return vi_sim


if __name__ == '__main__':
    
    # read Sri Lanka gpis
    #gpi_path = "C:\\Users\\i.pfeil\\Desktop\\Isabella\\pointlist_India_warp.csv"
    #gpis_df = pd.read_csv(gpi_path)
    #ind = np.where(gpis_df['cell']==1821)[0]
    #gpis1821 = gpis_df['point'].values[ind]
    
    # set paths to datasets
    ssm_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\ssm\\foxy_finn\\R1A\\"
    lcpath = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\LC\\ESACCI-LC-L4-LCCS-Map-300m-P5Y-20100101-West_SA-v1.6.1.nc"
    ndvi_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\NDVI\\NDVI_stack.nc"
    ndvi300_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\NDVI300\\"
    lai_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\LAI\\"
    swi_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\SWI\\SWI_stack.nc"
    fapar_path = "C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\FAPAR\\"
    
    paths = {'ssm': ssm_path, 'lc': lcpath, 'NDVI300': ndvi300_path, 
             'NDVI': ndvi_path, 'LAI': lai_path, 'SWI': swi_path, 
             'FAPAR': fapar_path}
    
    #===========================================================================
    # lon = 73.8
    # lat = 21
    # start_date = datetime(2007,7,1)
    # end_date = datetime(2015,7,1)
    # zribi(paths, lon, lat, start_date, end_date, t_val='SWI_020', vi_str='NDVI',
    #       plot_fig=False, monthly=False)
    #===========================================================================