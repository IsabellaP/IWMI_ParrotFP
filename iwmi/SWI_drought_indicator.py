from simon import anomaly, read_ts_area, plot_anomaly
from simon import init_0_1_grid, calc_IMD_10
from readers import read_ts, read_img
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import numpy as np
from simon import create_drought_dist
from simon import plot_Droughts_and_Anomalies
import pandas as pd
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import os
import matplotlib.gridspec as gridspec
from poets.shape.shapes import Shape
import pytesmo.temporal_matching as temp_match
import pytesmo.metrics as metrics

def drought_index(swi_path, param, lat_min, lat_max, lon_min, lon_max):
    swi_grid = init_0_1_grid('SWI')
    lat, lon = swi_grid.get_bbox_grid_points(lat_min, lat_max, lon_min, lon_max, coords=True)
    gp = swi_grid.get_bbox_grid_points(lat_min, lat_max, lon_min, lon_max)

    startdate = datetime(2007, 7, 1)
    enddate = datetime(2016, 7, 1)
    dataframe = pd.DataFrame()
    anomalies = pd.DataFrame()

    for ind in range(lat.size):
        swi = read_ts(swi_path, params=[param], lon=lon[ind], lat=lat[ind], start_date=startdate, end_date=enddate)
        anom = anomaly(swi)
        anom.columns = [gp[ind]]
        df = set_thresholds(anom)
        df.columns = [gp[ind]]
        dataframe = pd.concat([dataframe, df], axis=1)
        anomalies = pd.concat([anomalies, anom], axis=1)

    return dataframe, anomalies, lon, lat

def set_thresholds(data):
    neg_high = -10
    neg_low = -5
    pos_low = 3
    pos_high = 9
    df = data.copy()
    df[data < neg_high] = 1
    df[(data >= neg_high) & (data < neg_low)] = 2
    df[(data >= neg_low) & (data < pos_low)] = 3
    df[(data >= pos_low) & (data < pos_high)] = 4
    df[data >= pos_high] = 5
    return df

def read_mask(lat_min, lat_max, lon_min, lon_max):
    fpath = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWI\\20070701"
    fname = "g2_BIOPAR_SWI10_200707010000_GLOBE_ASCAT_V3_0_1.nc"
    with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]
        lat_idx1 = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx1 = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        data = (ncfile.variables["SWI_010"][lat_idx1, lon_idx1])
        if np.ma.is_masked(data):
            mask = (ncfile.variables["SWI_010"][lat_idx1, lon_idx1]).mask
        else:
            mask = None

    with Dataset('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\AG_Mask\\AG_LC_West_SA_0.1.nc', mode='r') as ncf:
        lons = ncf.variables['lon'][:]
        lats = ncf.variables['lat'][:]
        lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
        lc_mask = ncf.variables['AG_LC_dataset'][0, lat_idx, lon_idx]

    lc_mask = np.ma.masked_where((lc_mask != 1), lc_mask)

    if mask == None:
        return lc_mask
    else:
        ag_mask = lc_mask[np.where(mask == False)]
        return ag_mask


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    return plt.cm.get_cmap(base_cmap, N)

def plot_Drought_indices(df_d, lon_d, lat_d, imd_anom, lon_imd, lat_imd):

    lc_mask = read_mask(15.604600000000119, 22.03099899999995, 72.65069500000004, 80.89215899999999)    # 19.204, 21, 74, 76.5754
    mask = lc_mask.data.flatten()
    mask[np.where(mask != 1)] = 0
    mask_ind = np.where(mask == 1)

    dates = df_d.index.values

    for i, date in enumerate(dates[1:]):
        date = pd.to_datetime(str(date)).strftime('%Y-%m-%d')
        date_before = pd.to_datetime(str(dates[i])).strftime('%Y-%m-%d')
        gs = gridspec.GridSpec(4, 4)
        fig = plt.figure(figsize=[30, 25])
        plt.suptitle(date, fontsize=30)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        map = Basemap(projection='cyl', llcrnrlon=72, llcrnrlat=15, urcrnrlat=22.6,
                      urcrnrlon=81)
        #map = Basemap(projection='cyl', llcrnrlon=73, llcrnrlat=18.204, urcrnrlat=22,
        #              urcrnrlon=77.5754)
        map.drawmapboundary()
        map.drawcountries()
        cmap = cls.ListedColormap(['#A20000', '#DE0000', '#E5E618', '#47A917', '#30740C'], name='from_list', N=None)
        map.scatter(np.array(lon_d).flatten()[mask_ind], np.array(lat_d).flatten()[mask_ind],
                    c=df_d.loc[date].values.flatten()[mask_ind], edgecolor='None',
                    marker='s', s=52, vmin=1, vmax=5, cmap=discrete_cmap(6, cmap)) # s=125 for Maharashtra
        map.readshapefile(os.path.join('C:\\', 'Users', 's.hochstoger', 'Desktop',
                                     '0_IWMI_DATASETS', 'shapefiles', 'IND_adm',
                                     'IND_adm1'), 'IN.MH.JN', linewidth=0.4)
        map.readshapefile('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\shapefiles\\Maharashtra', 'IN.MH',
                          linewidth=3)
        cbar=plt.colorbar(ticks=range(6))
        cbar.ax.tick_params(labelsize=22)

        plt.clim(0.5, 6 - 0.5)
        plt.title("SWADI", fontsize=26)

        #IDSI
        data, lon, lat, _ = read_img('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\IDSI_stack.nc', 'IDSI',
                                     15.605, 22.031, 72.651, 80.893, timestamp=datetime.strptime(date, '%Y-%m-%d'))
        lons, lats = np.meshgrid(lon, lat)
        data[np.where(data == 0)] = 8
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        map = Basemap(projection='cyl', llcrnrlon=72, llcrnrlat=15, urcrnrlat=22.6,
                      urcrnrlon=81)
        map.drawmapboundary()
        map.drawcountries()

        cmap = cls.ListedColormap(['#A20000', '#DE0000', '#E19800', '#FBD37F', '#E5E618', '#47A917', '#30740C', 'w', 'b'], name='from_list', N=None)
        map.scatter(np.array(lons).flatten(), np.array(lats).flatten(), c=np.array(data).flatten(),
                            edgecolor='None', marker='s', s=0.1, vmin=1, vmax=9, cmap=discrete_cmap(10, cmap)) # s=0.3 for Maharashtra
        map.readshapefile(os.path.join('C:\\', 'Users', 's.hochstoger', 'Desktop',
                                     '0_IWMI_DATASETS', 'shapefiles', 'IND_adm',
                                     'IND_adm1'), 'IN.MH.JN', linewidth=0.4)
        map.readshapefile('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\shapefiles\\Maharashtra', 'IN.MH',
                          linewidth=3)
        cbar=plt.colorbar(ticks=range(10))
        cbar.ax.tick_params(labelsize=22)

        plt.clim(0.5, 10 - 0.5)
        plt.title("IDSI", fontsize=26)

        # Rainfall
        ax3 = fig.add_subplot(gs[2:4, 2:4])
        map = Basemap(projection='cyl', llcrnrlon=72, llcrnrlat=15, urcrnrlat=22.6,
                      urcrnrlon=81)
        map.drawmapboundary()
        map.drawcountries()

        cmap = 'RdBu'
        map.scatter(np.array(lon_imd).flatten()[mask_ind], np.array(lat_imd).flatten()[mask_ind],
                    c=imd_anom.loc[date_before].values.flatten()[mask_ind], edgecolor='None',
                    marker='s', s=52, vmin=-8, vmax=8, cmap=cmap)
        map.readshapefile(os.path.join('C:\\', 'Users', 's.hochstoger', 'Desktop',
                                     '0_IWMI_DATASETS', 'shapefiles', 'IND_adm',
                                     'IND_adm1'), 'IN.MH.JN', linewidth=0.4)
        map.readshapefile('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\shapefiles\\Maharashtra', 'IN.MH',
                          linewidth=3)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=22)

        plt.title("Rainfall Anomalies " + date_before, fontsize=26)

        ax4 = fig.add_subplot(gs[2:4, 0:2])
        map = Basemap(projection='cyl', llcrnrlon=72, llcrnrlat=15, urcrnrlat=22.6,
                      urcrnrlon=81)
        map.drawmapboundary()
        map.drawcountries()

        cmap = 'RdBu'
        map.scatter(np.array(lon_imd).flatten()[mask_ind], np.array(lat_imd).flatten()[mask_ind],
                    c=imd_anom.loc[date].values.flatten()[mask_ind], edgecolor='None',
                    marker='s', s=52, vmin=-8, vmax=8, cmap=cmap)
        map.readshapefile(os.path.join('C:\\', 'Users', 's.hochstoger', 'Desktop',
                                     '0_IWMI_DATASETS', 'shapefiles', 'IND_adm',
                                     'IND_adm1'), 'IN.MH.JN', linewidth=0.4)
        map.readshapefile('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\shapefiles\\Maharashtra', 'IN.MH',
                          linewidth=3)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=22)

        plt.title("Rainfall Anomalies " + date, fontsize=26)
        plt.savefig('C:\\Users\\s.hochstoger\\Desktop\\Plots\\SWI_IDSI_Drought\\SWI_IDSI_RF_Drought_MA_' + date + '.png', dpi=250,
                     bbox_inches='tight', pad_inches=0.3)
        plt.close()

def IMD_RF_10d_anomalies(lat_min, lat_max, lon_min, lon_max):
    imd_path = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\IMD_RF_stack.nc'
    grid = init_0_1_grid('SWI')
    lats, lons = grid.get_bbox_grid_points(lat_min, lat_max, lon_min, lon_max, coords=True)
    gp = grid.get_bbox_grid_points(lat_min, lat_max, lon_min, lon_max)
    startdate = datetime(2007, 1, 1)
    enddate = datetime(2015, 12, 31)

    # IMD RAINFALL ANOMALIES
    dataframe = pd.DataFrame()
    anomalies = pd.DataFrame()
    for ind in range(lats.size):
        rf = read_ts(imd_path, ['IMD_RF'], lons[ind], lats[ind], startdate, enddate)
        df = calc_IMD_10(rf)
        anom = anomaly(df)
        df.columns = [gp[ind]]
        anom.columns = [gp[ind]]
        dataframe = pd.concat([dataframe, df], axis=1)
        anomalies = pd.concat([anomalies, anom], axis=1)
    return anomalies, dataframe

def create_SWADI_dist(swi_path, lat_min, lat_max, lon_min, lon_max, df=None):
    if df is None:
        df_d, lon, lat = drought_index(swi_path, lat_min, lat_max, lon_min, lon_max)
    else:
        df_d = df
    columns = ['healthy', 'watch', 'drought']
    df = pd.DataFrame([], index=[-999], columns=columns)

    lc_mask = read_mask(lat_min, lat_max, lon_min, lon_max)
    mask = lc_mask.data.flatten()
    mask[np.where(mask != 1)] = 0
    mask_ind = np.where(mask == 1)

    df_d = df_d.iloc[:, mask_ind[0]]

    dates = df_d.index
    for day in dates:
        drought = np.where((df_d.loc[day] == 1) | (df_d.loc[day] == 2))[0].size
        watch = np.where((df_d.loc[day] == 3))[0].size
        healthy = np.where((df_d.loc[day] == 4) | (df_d.loc[day] == 5))[0].size

        mean = pd.DataFrame([(healthy, watch, drought)], index=[day], columns=df.columns)
        df = df.append(mean)

    df = df[df.index != -999]
    df.index = pd.to_datetime(dates)
    df = df.divide(df.sum(axis=1), axis=0)
    df.drought = df.drought * (-1)

    return df

def get_lonlat_district(string):
    region = string
    # districts
    shapefile = os.path.join('C:\\', 'Users', 's.hochstoger', 'Desktop',
                             '0_IWMI_DATASETS', 'shapefiles', 'IND_adm',
                             'IND_adm2')

    shpfile = Shape(region, shapefile=shapefile)
    lon_min, lat_min, lon_max, lat_max = shpfile.bbox
    return lat_min, lat_max, lon_min, lon_max

if __name__ == '__main__':

    swi_path = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\SWI_stack.nc'
    lat_min, lat_max, lon_min, lon_max = get_lonlat_district('IN.MH.AU') #IN.GJ.AM - IN.RJ.BP



    #swi001 = read_ts_area(swi_path, 'SWI_001', lat_min, lat_max, lon_min, lon_max)
    # swi040 = read_ts_area(swi_path, 'SWI_040', lat_min, lat_max, lon_min, lon_max)
    # swi_anom = anomaly(swi040)
    # plot_anomaly(swi040, swi_anom)


    # ============== SWI and rainfall correlations for each GP
    # imd_path = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\IMD_RF_stack.nc'
    # grid = init_0_1_grid('SWI')
    # lats, lons = grid.get_bbox_grid_points(lat_min, lat_max, lon_min, lon_max, coords=True)
    # startdate = datetime(2007, 7, 1)
    # enddate = datetime(2015, 12, 31)
    # corr = np.array([])
    #
    # for ind in range(lats.size):
    #     rf = read_ts(imd_path, ['IMD_RF'], lons[ind], lats[ind], startdate, enddate)
    #     swi = read_ts(swi_path, ['SWI_001'], lons[ind], lats[ind], startdate, enddate)
    #     df = calc_IMD_10(rf)
    #     match = temp_match.matching(swi, rf)
    #     s_rho, s_p = metrics.spearmanr(match.iloc[:, 0], match.iloc[:, 1])
    #
    #     corr = np.append(corr, s_rho)
    # ========================================

    IDSI = create_drought_dist(lat_min, lat_max, lon_min, lon_max)
    IDSI.drought = IDSI.drought * (-1)
    SWADI, _, _ = create_SWADI_dist(swi_path, lat_min, lat_max, lon_min, lon_max)

    rf_anom, rf = IMD_RF_10d_anomalies(lat_min, lat_max, lon_min, lon_max)

    #=========== load csv files for whole MA =========
    # IDSI = create_drought_dist(lat_min, lat_max, lon_min, lon_max)
    # IDSI.drought = IDSI.drought * (-1)
    #
    # SWADI = pd.read_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWADI_data\\All_Maharashtra\\SWADI_Maharashtra_10050309.csv")
    # SWADI.index = SWADI.iloc[:, 0].values
    # SWADI = SWADI.drop('Unnamed: 0', 1)
    # SWADI.index = pd.to_datetime(SWADI.index)
    # lonlat = pd.read_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWADI_data\\All_Maharashtra\\SWADI_lon_lat_Maharashtra_10050309.csv')
    # lon_d = lonlat.lon.values
    # lat_d = lonlat.lat.values
    # SWADI_mask = SWADI.iloc[:, mask_ind[0]]
    #
    # SWADI = create_SWADI_dist(lat_min, lat_max, lon_min, lon_max, df=SWADI_mask)
    #
    # rf_anom = pd.read_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWADI_data\\All_Maharashtra\\IMD_rainfall_anomaly_Maharashtra.csv')
    # rf_anom.index = rf_anom.iloc[:, 0].values
    # rf_anom = rf_anom.drop('Unnamed: 0', 1)
    # rf_anom.index = pd.to_datetime(rf_anom.index)
    # rf = pd.read_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWADI_data\\All_Maharashtra\\IMD_rainfall_Maharashtra.csv')
    # rf.index = rf.iloc[:, 0].values
    # rf = rf.drop('Unnamed: 0', 1)
    # rf.index = pd.to_datetime(rf.index)
    # =========================================

    rf_anom_mean = pd.DataFrame(rf_anom.mean(axis=1)/(rf_anom.mean(axis=1).max()))
    rf_anom_mean = rf_anom_mean.loc['2007-07-01':'2015-06-26']
    rf_mean = rf.mean(axis=1).loc['2007-07-01':'2015-06-26']
    rf_anom_mean.columns = ['Rainfall Anomalies']
    rf_mean.columns = ['Rainfall']

    #plot_Drought_indices(SWADI, lon_d, lat_d, rf_anom, lon_d, lat_d)

    #match = temp_match.matching(swi040, rf_mean)
    #s_rho, s_p = metrics.spearmanr(match.iloc[:, 0], match.iloc[:, 1])

    #============= Correlation Bias RMSD for SWADI for different thresholds
    IDSI_drought = pd.DataFrame((IDSI.drought).loc['2007-07-01':])
    SWADI_drought = pd.DataFrame((SWADI.drought).loc[:'2015-07-01'])
    IDSI_drought.columns = ['IDSI Drought']
    SWADI_drought.columns = ['SWADI Drought']

    match = temp_match.matching(SWADI_drought, IDSI_drought)
    s_rho, s_p = metrics.spearmanr(match.iloc[:, 0], match.iloc[:, 1])
    BIAS = metrics.bias(match.iloc[:, 0], match.iloc[:, 1])
    RMSD = metrics.rmsd(match.iloc[:, 0], match.iloc[:, 1])

    IDSI_healthy = pd.DataFrame((IDSI.healthy).loc['2007-07-01':])
    SWADI_healthy = pd.DataFrame((SWADI.healthy).loc[:'2015-07-01'])
    IDSI_healthy.columns = ['IDSI healthy']
    SWADI_healthy.columns = ['SWADI healthy']

    match_h = temp_match.matching(SWADI_healthy, IDSI_healthy)
    s_rho_h, s_p_h = metrics.spearmanr(match_h.iloc[:, 0], match_h.iloc[:, 1])
    BIAS_h = metrics.bias(match_h.iloc[:, 0], match_h.iloc[:, 1])
    RMSD_h = metrics.rmsd(match_h.iloc[:, 0], match_h.iloc[:, 1])

    IDSI_watch = pd.DataFrame((IDSI.watch).loc['2007-07-01':])
    SWADI_watch = pd.DataFrame((SWADI.watch).loc[:'2015-07-01'])
    IDSI_watch.columns = ['IDSI healthy']
    SWADI_watch.columns = ['SWADI healthy']

    match_w = temp_match.matching(SWADI_watch, IDSI_watch)
    s_rho_w, s_p_w = metrics.spearmanr(match_w.iloc[:, 0], match_w.iloc[:, 1])
    BIAS_w = metrics.bias(match_w.iloc[:, 0], match_w.iloc[:, 1])
    RMSD_w = metrics.rmsd(match_w.iloc[:, 0], match_w.iloc[:, 1])

    drought_stat = [s_rho, s_p, BIAS, RMSD]
    healthy_stat = [s_rho_h, s_p_h, BIAS_h, RMSD_h]
    watch_stat = [s_rho_w, s_p_w, BIAS_w, RMSD_w]
    stat_all = np.array([drought_stat, healthy_stat, watch_stat])

    df = pd.DataFrame(stat_all, index=['drought', 'healthy', 'watch'], columns=['R', 'p', 'BIAS', 'RMSD'])
    df.to_csv('C:\\Users\\s.hochstoger\\Desktop\\Plots\\SWADI_IDSI_STATS\\AU_10050309_SWI40_stats.csv')
    # =================================================

    #============== plot IDSI vs SWADI and IMD Rainfall
    fig = plt.figure(figsize=[30, 15])
    plt.suptitle("IDSI vs. SWADI in Maharashtra", fontsize=22)
    ax1 = fig.add_subplot(211)
    IDSI.loc['2007-07-01':'2015-06-26'].plot.area(ax=ax1, alpha=0.4, color='gyr')
    rf_anom_mean.plot.area(ax=ax1, alpha=0.4, stacked=False)
    ax1.set_title('IDSI', fontsize=18)
    plt.grid()
    plt.axhline(0, color='black')
    plt.ylim([-1.1, 1.1])
    ax1.set_xticks(SWADI.index[::18][:-2])
    ax1.set_xticklabels(SWADI.index[::18][0:-2].date, fontsize=12)

    ax2 = fig.add_subplot(212)
    SWADI.loc['2007-07-01':'2015-07-01'].plot.area(ax=ax2, alpha=0.4, color='gyr')
    rf_anom_mean.plot.area(ax=ax2, alpha=0.4, stacked=False)
    ax2.set_title('SWADI', fontsize=18)
    plt.grid()
    plt.axhline(0, color='black')
    plt.ylim([-1.1, 1.1])
    ax2.set_xticks(SWADI.index[::18][:-2])
    ax2.set_xticklabels(SWADI.index[::18][0:-2].date, fontsize=12)
    # plt.savefig('C:\\Users\\s.hochstoger\\Desktop\\Plots\\SWADI_IDSI_STATS\\JN_10040209.png',
    #            dpi=250, bbox_inches='tight', pad_inches=0.3)

    plt.figure(figsize=[30, 7])
    rf_mean.plot()
    plt.grid()
    plt.title('IMD Rainfall 10-daily', fontsize=18)
    plt.show()
    # ===========================================

    # ===================== plot IDSI, SWADI and RF anomalies in one plot
    #SWI drought
    # df_d, lon_d, lat_d = drought_index(14.7148, 29.3655, 68.15, 81.8419)
    # df_d.to_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\test_drougth_all.csv")
    #
    # d = {'lon': lon_d, 'lat': lat_d}
    # lonlat = pd.DataFrame(data=d)
    # lonlat.to_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\test_drought_lon_lat_all.csv')

    # df_d = pd.read_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWADI_data\\test_drougth_all.csv")
    # df_d.index = df_d.iloc[:, 0].values
    # df_d = df_d.drop('Unnamed: 0', 1)
    # lonlat = pd.read_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWADI_data\\test_drought_lon_lat_all.csv')
    # lon_d = lonlat.lon.values
    # lat_d = lonlat.lat.values
    # del_ind = np.where(lon_d != 68.15)
    # lon_d = lon_d[del_ind]
    # lat_d = lat_d[del_ind]
    # df_d = df_d.iloc[:, del_ind[0]]
    #
    # imd_anom = pd.read_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWADI_data\\IMD_anom_all.csv")
    # imd_anom.index = imd_anom.iloc[:, 0].values
    # imd_anom = imd_anom.drop('Unnamed: 0', 1)
    # imd_lonlat = pd.read_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWADI_data\\IMD_lon_lat_all.csv')
    # lon_imd = imd_lonlat.lon.values
    # lat_imd = imd_lonlat.lat.values
    # del_ind = np.where(lon_imd != 68.15)
    # lon_imd = lon_imd[del_ind]
    # lat_imd = lat_imd[del_ind]
    # imd_anom = imd_anom.iloc[:, del_ind[0]]
    #
    # plot_Drought_indices(df_d, lon_d, lat_d, imd_anom, lon_imd, lat_imd)
    # ========================================================

    pass

