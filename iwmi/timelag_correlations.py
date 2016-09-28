import os
import numpy as np
import pandas as pd
from datetime import datetime
#from data_readers import init_0_4_grid
from data_analysis import corr, plot_max_timelags
from poets.shape.shapes import Shape
from netCDF4 import Dataset

# set paths to datasets
swi_path = "E:\\poets\\RAWDATA\\SWI_daily\\SWI_daily_stack.nc"
ndvi_path = "E:\\poets\\RAWDATA\\NDVI_stack\\NDVI_gapfree.nc"

paths = {'SWI': swi_path, 'NDVI': ndvi_path}
 
# poets lonlat
#grid = init_0_4_grid()
#gpis, lons, lats = grid.get_grid_points()

# gpis for one district with 0.1 degree resolution
with Dataset(swi_path, 'r') as ncfile:
    res_lons = ncfile.variables['lon'][:]
    res_lats = ncfile.variables['lat'][:]
shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                             '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                             'IND_adm2')
region = 'IN.MH.JN'
shpfile = Shape(region, shapefile=shapefile)
lon_min, lat_min, lon_max, lat_max = shpfile.bbox
lons = res_lons[np.where((res_lons>=lon_min) & (res_lons<=lon_max))]
lats = res_lats[np.where((res_lats>=lat_min) & (res_lats<=lat_max))]
print lons.shape, lats.shape

#plot_max_timelags(lons, lats)
#plot_corr_new(lons, lats)

start_date = datetime(2007, 7, 1)
end_date = datetime(2015, 7, 1)
max_rho = {}
time_lags = [0, 8, 16, 24, 32]
corr_df = pd.DataFrame([], index=time_lags)
   
max_corr_val = []
max_corr_swi = []
max_corr_lag = []

for lon in lons:
    for lat in lats:
        print lon, lat
        corr_df = corr(paths, corr_df, start_date, end_date, 
                       lon=lon, lat=lat,
                       vi_str='NDVI', time_lags=time_lags)
          
        #print corr_df
        #max_rho = max_corr(corr_df, max_rho)
        corr_rho = corr_df[['NDVI_SWI_001_rho', 'NDVI_SWI_010_rho',
                             'NDVI_SWI_020_rho', 'NDVI_SWI_040_rho',
                             'NDVI_SWI_060_rho', 'NDVI_SWI_100_rho']]
        max_corr_val.append(corr_rho.max(axis=0).max())
        max_corr_swi.append(corr_rho.max(axis=0).idxmax())
        max_corr_lag.append(corr_rho.max(axis=1).idxmax())
     
max_corr_val = np.array(max_corr_val)
max_corr_swi = np.array(max_corr_swi)
max_corr_lag = np.array(max_corr_lag)
    
np.save('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_corr_val2.npy', 
        max_corr_val)
np.save('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_corr_swi2.npy', 
        max_corr_swi)
np.save('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_corr_lag2.npy', 
        max_corr_lag)

plot_max_timelags(lons, lats)

