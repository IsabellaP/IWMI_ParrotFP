from simon import anomaly
from simon import init_0_1_grid
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

def drought_index(lat_min, lat_max, lon_min, lon_max):
    swi_grid = init_0_1_grid('SWI')
    lat, lon = swi_grid.get_bbox_grid_points(lat_min, lat_max, lon_min, lon_max, coords=True)
    gp = swi_grid.get_bbox_grid_points(lat_min, lat_max, lon_min, lon_max)

    startdate = datetime(2007, 7, 1)
    enddate = datetime(2016, 7, 1)
    dataframe = pd.DataFrame()

    for ind in range(lat.size):
        swi = read_ts(swi_path, params=['SWI_040'], lon=lon[ind], lat=lat[ind], start_date=startdate, end_date=enddate)
        anom = anomaly(swi)
        anom = anom/100
        df = set_thresholds(anom)
        df.columns = [gp[ind]]
        dataframe = pd.concat([dataframe, df], axis=1)

        if ind % 500 == 0:
            print ind

    return dataframe, lon, lat

def set_thresholds(data):
    df = data.copy()
    df[data < -0.09] = 1
    df[(data >= -0.09) & (data < -0.03)] = 2
    df[(data >= -0.03) & (data < 0.03)] = 3
    df[(data >= 0.03) & (data < 0.09)] = 4
    df[data >= 0.09] = 5
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
        mask = (ncfile.variables["SWI_010"][lat_idx1, lon_idx1]).mask

    #grid = BasicGrid(lons[np.where(mask == False)], lats[np.where(mask == False)])

    with Dataset('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\AG_Mask\\AG_LC_West_SA_0.1.nc', mode='r') as ncf:
        lons = ncf.variables['lon'][:]
        lats = ncf.variables['lat'][:]
        lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
        lc_mask = ncf.variables['AG_LC_dataset'][0, lat_idx, lon_idx]

    lc_mask = np.ma.masked_where((lc_mask != 1), lc_mask)
    ag_mask = lc_mask[np.where(mask == False)]
    return ag_mask


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    return plt.cm.get_cmap(base_cmap, N)

if __name__ == '__main__':
    swi_path = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\SWI_stack.nc'
    lon_min = 74
    lon_max = 76.5754
    lat_min = 19.204
    lat_max = 21

    lc_mask = read_mask(14.7148, 29.3655, 68.25, 81.8419)
    mask = lc_mask.data.flatten()
    mask[np.where(mask != 1)] = 0
    mask_ind = np.where(mask == 1)

    #SWI drought
    #df_d, lon_d, lat_d = drought_index(14.7148, 29.3655, 68.15, 81.8419)
    #df_d.to_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\test_drougth_all.csv")

    #d = {'lon': lon_d, 'lat': lat_d}
    #lonlat = pd.DataFrame(data=d)
    #lonlat.to_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\test_drought_lon_lat_all.csv')

    df_d = pd.read_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\test_drougth_all.csv")
    df_d.index = df_d.iloc[:, 0].values
    df_d = df_d.drop('Unnamed: 0', 1)
    lonlat = pd.read_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\test_drought_lon_lat_all.csv')
    lon_d = lonlat.lon.values
    lat_d = lonlat.lat.values
    del_ind = np.where(lon_d != 68.15)
    lon_d = lon_d[del_ind]
    lat_d = lat_d[del_ind]
    df_d = df_d.iloc[:, del_ind[0]]

    # columns = ['healthy', 'watch', 'drought']
    # df = pd.DataFrame([], index=[-999], columns=columns)
    #
    # dates = df_d.index
    # for day in dates:
    #     drought = np.where((df_d.loc[day] == 1) | (df_d.loc[day] == 2))[0].size
    #     watch = np.where((df_d.loc[day] == 3))[0].size
    #     healthy = np.where((df_d.loc[day] == 4) | (df_d.loc[day] == 5))[0].size
    #
    #     mean = pd.DataFrame([(healthy, watch, drought)], index=[day], columns=df.columns)
    #     df = df.append(mean)
    #
    # df = df[df.index != -999]
    # df.index = dates
    # df = df.divide(df.sum(axis=1), axis=0)


    dates = df_d.index.values

    for date in dates[41:]:
        fig = plt.figure(figsize=[30, 8])
        plt.suptitle(date, fontsize=20)
        ax1 = fig.add_subplot(121)
        map = Basemap(projection='cyl', llcrnrlon=68.145, llcrnrlat=14.61, urcrnrlat=29.47,
                      urcrnrlon=81.848)
        map.drawmapboundary()
        #map.drawcountries()
        cmap = cls.ListedColormap(['#A20000', '#DE0000', '#E5E618', '#47A917', '#30740C'], name='from_list', N=None)
        map.scatter(np.array(lon_d).flatten()[mask_ind], np.array(lat_d).flatten()[mask_ind],
                    c=df_d.loc[date].values.flatten()[mask_ind], edgecolor='None',
                    marker='s', s=10, vmin=1, vmax=5, cmap=discrete_cmap(6, cmap))
        map.readshapefile(os.path.join('C:\\', 'Users', 's.hochstoger', 'Desktop',
                                     '0_IWMI_DATASETS', 'shapefiles', 'IND_adm',
                                     'IND_adm1'), 'IN.MH.JN')
        plt.colorbar(ticks=range(6))
        plt.clim(0.5, 6 - 0.5)
        plt.title("SWI Anomaly Index")

        #IDSI
        data, lon, lat = read_img('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\IDSI_stack.nc', 'IDSI',
                                  14.7148, 29.3655, 68.15, 81.8419, timestamp=datetime.strptime(date, '%Y-%m-%d'))
        lons, lats = np.meshgrid(lon, lat)
        data[np.where(data == 0)] = 8
        ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
        map = Basemap(projection='cyl', llcrnrlon=68.145, llcrnrlat=14.71, urcrnrlat=29.37,
                      urcrnrlon=81.848)
        map.drawmapboundary()
        #map.drawcountries()

        cmap = cls.ListedColormap(['#A20000', '#DE0000', '#E19800', '#FBD37F', '#E5E618', '#47A917', '#30740C', 'w', 'b'], name='from_list', N=None)
        map.scatter(np.array(lons).flatten(), np.array(lats).flatten(), c=np.array(data).flatten(),
                            edgecolor='None', marker='s', s=0.03, vmin=1, vmax=9, cmap=discrete_cmap(10, cmap))
        map.readshapefile(os.path.join('C:\\', 'Users', 's.hochstoger', 'Desktop',
                                     '0_IWMI_DATASETS', 'shapefiles', 'IND_adm',
                                     'IND_adm1'), 'IN.MH.JN')
        plt.colorbar(ticks=range(10))
        plt.clim(0.5, 10 - 0.5)
        plt.title("IDSI")
        plt.savefig('C:\\Users\\s.hochstoger\\Desktop\\Plots\\SWI_IDSI_Drought\\SWI_IDSI_Drought_' + date + '.png', dpi=450,
                    bbox_inches='tight', pad_inches=0.3)
        plt.close()

    pass

