from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from veg_pred_readers import read_ts_area
from datetime import datetime

vi_path = 'E:\\poets\\RAWDATA\\NDVI_stack\\NDVI_gapfree.nc'
#===============================================================================
# swi_path = 'E:\\poets\\RAWDATA\\SWI_daily_stack\\SWI_daily_stack.nc'
# 
# with Dataset(vi_path, 'r') as ncfile:
#     for var in ncfile.variables.keys():
#         if not 'NDVI' in var:
#             continue
#         data = ncfile.variables[var][:,3516,4834]
#         nctime = ncfile.variables['time'][:]
#         unit_temps = ncfile.variables['time'].units
#         try:
#             cal_temps = ncfile.variables['time'].calendar
#         except AttributeError:  # Attribute doesn't exist
#             cal_temps = u"gregorian"  # or standard
#     
#         nc_all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)
#     
#         plt.plot(nc_all_dates, data)
#         plt.show()
# 
# print 'done'
#===============================================================================

lon = 75.65
lat = 19.95
start_date = datetime(2007,1,1)
end_date = datetime(2016,10,1)
vi_all = read_ts_area(vi_path, 'NDVI', 
                      lat_min=lat-0.05, lat_max=lat+0.05, 
                      lon_min=lon-0.05, lon_max=lon+0.05)
vi_df = vi_all[start_date:end_date]

pd.set_option('display.max_rows', None)
print vi_df

pred1 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\IN.MH.JN_20130602.npy')
pred2 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\IN.MH.JN_20130610.npy')
pred3 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\IN.MH.JN_20130618.npy')
pred4 = np.load('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\02_results\\IN.MH.JN_20130626.npy')

idx1 = np.where((np.around(pred1[:,0].astype(np.double),2)==lon) & (np.around(pred1[:,1].astype(np.double),2)==lat))[0][0]
idx2 = np.where((np.around(pred2[:,0].astype(np.double),2)==lon) & (np.around(pred2[:,1].astype(np.double),2)==lat))[0][0]
idx3 = np.where((np.around(pred3[:,0].astype(np.double),2)==lon) & (np.around(pred3[:,1].astype(np.double),2)==lat))[0][0]
idx4 = np.where((np.around(pred4[:,0].astype(np.double),2)==lon) & (np.around(pred4[:,1].astype(np.double),2)==lat))[0][0]

print idx1, idx2, idx3, idx4

df = pd.DataFrame([pred1[idx1,2], pred2[idx2,2], pred3[idx3,2], pred4[idx4,2]],
             columns=['NDVI'], index=pd.Series([pred1[idx1,3], pred2[idx2,3],
                                      pred3[idx3,3], pred4[idx4,3]]))

print df 

print 'done'
