from poets.shape.shapes import Shape
from veg_pred_readers import read_ts_area
import matplotlib.pyplot as plt
import numpy as np


def anomaly(df):
    '''
    Calculates anomalies for time series. Of each day mean value of
    this day over all years is subtracted.
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    Returns:
    --------
    data : pd.DataFrame
        Dataset containing anomalies of input DataFrame
    '''
    group = df.groupby([df.index.month, df.index.day])
    m = {}
    df_anomaly = df.copy()
    clima = df.copy()
    for key, _ in group:
        m[key] = group.get_group(key).mean()

    for i in range(0, len(df_anomaly)):
        val = m[(df_anomaly.index[i].month, df_anomaly.index[i].day)]
        df_anomaly.iloc[i] = df_anomaly.iloc[i] - val
        clima.iloc[i] = val

    col_str = df.columns[0] + ' Anomaly'
    df_anomaly.columns = [col_str]

    return df_anomaly, clima


swi_path = 'E:\\poets\\RAWDATA\\SWI_daily_stack\\SWI_daily_stack.nc'
vi_path = 'E:\\poets\\RAWDATA\\NDVI_stack\\NDVI_gapfree.nc'
shp_path = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\shapefiles\\IND_adm\\IND_adm2.shp'
region = 'IN.MH.JN'
shpfile = Shape(region, shapefile=shp_path)
lon_min, lat_min, lon_max, lat_max = shpfile.bbox

vi_df = read_ts_area(vi_path, 'NDVI', 
                      lat_min=lat_min, lat_max=lat_max, 
                      lon_min=lon_min, lon_max=lon_max)

vi_clima = vi_df.groupby([vi_df.index.month]).mean()

swi_df = read_ts_area(swi_path, 'SWI_040', 
                      lat_min=lat_min, lat_max=lat_max, 
                      lon_min=lon_min, lon_max=lon_max)

swi_clima = swi_df.groupby([swi_df.index.month]).mean()

#===============================================================================
# _, vi_clima = anomaly(vi_df)
# _, swi_clima = anomaly(swi_df)
#===============================================================================
swi_resc = (swi_clima - swi_clima.min()) / (swi_clima.max() - swi_clima.min()) * 100
vi_resc = (vi_clima - vi_clima.min()) / (vi_clima.max() - vi_clima.min()) * 100

ax = vi_resc.plot()
swi_resc.plot(ax=ax)
plt.title('SWI and NDVI climatology - Jalna (MH)', fontsize=20)
plt.xticks(range(1,13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 
            'Sep', 'Oct', 'Nov', 'Dec'], rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Min-Max Rescaled SWI and NDVI', fontsize=18)
plt.show()

print 'done'