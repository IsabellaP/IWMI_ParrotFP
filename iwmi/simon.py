import os
import numpy as np
import pandas as pd
from datetime import datetime
from data_readers import read_img
from pygrids.warp5 import DGGv21CPv20
from rsdata.WARP.interface import WARP
from warp_data.interface import init_grid
import matplotlib.pyplot as plt
import pytesmo.scaling as scaling
import pytesmo.temporal_matching as temp_match
import pytesmo.metrics as metrics
from pytesmo.time_series.anomaly import calc_anomaly, calc_climatology

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


def plot_anomaly(df):

    if df.columns[0][0:3] == 'SWI':
        df = df/100

    df.plot.area(stacked=False)
    plt.grid()
    plt.axhline(0, color='black')


if __name__ == '__main__':

    ndvi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\VIs\\NDVI\\"
    swi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWI\\"

    lat_min = 16
    lat_max = 22
    lon_min = 70
    lon_max = 77

    df_ndvi = read_ts_area(ndvi_path, "NDVI", lat_min, lat_max, lon_min, lon_max)
    df_swi = read_ts_area(swi_path, "SWI", lat_min, lat_max, lon_min, lon_max, 20)

    anomaly_ndvi = anomaly(df_ndvi)
    anomaly_swi = anomaly(df_swi)

    plot_anomaly(anomaly_ndvi)
    plot_anomaly(anomaly_swi)
    plt.show()
    pass
