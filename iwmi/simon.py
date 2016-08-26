import os
import numpy as np
import pandas as pd
from datetime import datetime
from data_readers import read_img
from netCDF4 import Dataset, num2date
from mpl_toolkits.basemap import Basemap
from pygrids.warp5 import DGGv21CPv20
from rsdata.WARP.interface import WARP
from warp_data.interface import init_grid
import matplotlib.pyplot as plt
import pytesmo.scaling as scaling
import pytesmo.temporal_matching as temp_match
import pytesmo.metrics as metrics
from pytesmo.time_series.anomaly import calc_anomaly, calc_climatology
from scipy import spatial

def read_ts_area(path, param, lat_min, lat_max, lon_min, lon_max, t=1):

    folders = os.listdir(path)
    swi = 'SWI_' + str(t).zfill(3)
    mean = []
    dates = []
    for day in folders:
        date = datetime.strptime(day, '%Y%m%d')
        data = read_img(path, param=param, lat_min=lat_min, lat_max=lat_max,
                        lon_min=lon_min, lon_max=lon_max, timestamp=date, plot_img=False,
                        swi=swi)
        #if np.where(ndvi.data == 255)[0].size > 25000:
        #    mean = np.NAN
        if np.ma.is_masked(data):
            mean_value = data.data[np.where(data.data != 255)].mean()
        else:
            mean_value = data.mean()
        mean.append(mean_value)
        dates.append(date)

    data_df = {param: mean}
    df = pd.DataFrame(data=data_df, index=dates)
    if df.columns == 'SWI':
        df.columns = [swi]

    return df


def anomaly(df):

    group = df.groupby([df.index.month, df.index.day])
    m = {}
    df_anomaly = df.copy()
    for key, _ in group:
        m[key] = group.get_group(key).mean()

    for i in range(0, len(df_anomaly)):
        val = m[(df_anomaly.index[i].month, df_anomaly.index[i].day)]
        df_anomaly.iloc[i] = df_anomaly.iloc[i] - val

    col_str = df.columns[0] + ' Anomaly'
    df_anomaly.columns = [col_str]

    return df_anomaly


def plot_anomaly(df, df_anom):

    if df_anom.columns[0][0:3] == 'SWI':
        df_anom = df_anom/100
        df = df/100
        ax = df_anom.plot.area(stacked=False, figsize=[20, 15], color='b')
        df.plot(ax=ax, color='b')
        plt.title(df.columns[0] + ' and ' + df_anom.columns[0])
    else:
        ax = df_anom.plot.area(stacked=False, figsize=[20, 15], color='g')
        df.plot(ax=ax, color='g')
        plt.title(df.columns[0] + ' and ' + df_anom.columns[0])
    plt.grid()
    plt.axhline(0, color='black')
    plt.ylim([-0.5, 1])

def plot_area(lon_min, lon_max, lat_min, lat_max):

    path_to_nc_cop = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\VIs\\NDVI\\20070703'
    fname_cop = 'g2_BIOPAR_NDVI_200707030000_ASIA_VGT_V1_3.nc'
    with Dataset(os.path.join(path_to_nc_cop, fname_cop), mode='r') as ncfile:
        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]

    grid = np.meshgrid(lons, lats)
    cop_grid_lons = grid[0].flatten()
    cop_grid_lats = grid[1].flatten()

    index = np.where((cop_grid_lats <= lat_max) &
                     (cop_grid_lats >= lat_min) &
                     (cop_grid_lons <= lon_max) &
                     (cop_grid_lons >= lon_min))

    map = Basemap(projection='cyl', llcrnrlon=lons.min(), llcrnrlat=lats.min(), urcrnrlat=lats.max(), urcrnrlon=lons.max())
    map.drawmapboundary()
    map.drawcoastlines()
    map.drawcountries()

    map.plot(cop_grid_lons[index], cop_grid_lats[index], marker='+', linewidth=0, color='m', markersize=5)



if __name__ == '__main__':

    ndvi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\VIs\\NDVI\\"
    swi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWI\\"

    lat_min = 19.20
    lat_max = 19.437
    lon_min = 74.86
    lon_max = 75.13

    plot_area(lon_min, lon_max, lat_min, lat_max)
    plt.savefig('C:\Users\s.hochstoger\Desktop\Plots\\' + 'area.png')

    df_ndvi = read_ts_area(ndvi_path, "NDVI", lat_min, lat_max, lon_min, lon_max)

    df_swi1 = read_ts_area(swi_path, "SWI", lat_min, lat_max, lon_min, lon_max, 1)
    df_swi5 = read_ts_area(swi_path, "SWI", lat_min, lat_max, lon_min, lon_max, 5)
    df_swi10 = read_ts_area(swi_path, "SWI", lat_min, lat_max, lon_min, lon_max, 10)
    df_swi15 = read_ts_area(swi_path, "SWI", lat_min, lat_max, lon_min, lon_max, 15)
    df_swi20 = read_ts_area(swi_path, "SWI", lat_min, lat_max, lon_min, lon_max, 20)
    df_swi40 = read_ts_area(swi_path, "SWI", lat_min, lat_max, lon_min, lon_max, 40)
    df_swi60 = read_ts_area(swi_path, "SWI", lat_min, lat_max, lon_min, lon_max, 60)
    df_swi100 = read_ts_area(swi_path, "SWI", lat_min, lat_max, lon_min, lon_max, 100)

    swis = [df_swi1, df_swi5, df_swi10, df_swi15, df_swi20, df_swi40, df_swi60, df_swi100]

    anomaly_ndvi = anomaly(df_ndvi)
    plot_anomaly(df_ndvi.loc[:'20140403'], anomaly_ndvi.loc[:'20140403'])
    plt.savefig('C:\Users\s.hochstoger\Desktop\Plots\\' + df_ndvi.columns[0] + '.png')

    for swi in swis:
        anomaly_swi = anomaly(swi)
        plot_anomaly(swi.loc[:'20140403'], anomaly_swi.loc[:'20140403'])
        plt.savefig('C:\Users\s.hochstoger\Desktop\Plots\\' + swi.columns[0] + '.png')

    plt.show()
    pass
