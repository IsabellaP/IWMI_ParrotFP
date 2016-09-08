import pandas as pd
from datetime import datetime
from data_readers import init_0_4_grid
from data_analysis import corr

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
# grid = init_SWI_grid()
# lons = grid.activearrlon
# lats = grid.activearrlat
#===========================================================================
  
#===========================================================================
# # shpfile-bbox
# lonlat_idx = np.where((lats>=14.7) & (lats<=29.4) & (lons>=68.0) & 
#                       (lons<=81.8))[0]
# lons_shp = lons[lonlat_idx]
# lats_shp = lats[lonlat_idx]
#===========================================================================
 
# poets lonlat
#grid = init_SWI_grid()
grid = init_0_4_grid()
gpis, lons, lats = grid.get_grid_points()
print gpis.shape

#plot_max_timelags(lons, lats)
#plot_corr_new(lons, lats)
  
start_date = datetime(2007, 7, 1)
end_date = datetime(2015, 7, 1)
max_rho = {}
time_lags = [0]#, 10, 20, 30, 40, 50, 60, 100]
corr_df = pd.DataFrame([], index=time_lags)
   
max_corr_val = []
max_corr_swi = []
max_corr_lag = []
 
swi001 = []
swi010 = []
swi020 = []
swi040 = []
swi060 = []
swi100 = []
          
for i in range(len(gpis)):
    print i
    lon2 = 75.560811
    lat2 = 20.275159
    corr_df = corr(paths, corr_df, start_date, end_date, 
                   #lon=lons[i], lat=lats[i],
                   lon=lon2, lat=lat2, 
                   vi_str='NDVI', time_lags=time_lags)
    
#===========================================================================
#     #print corr_df
#     #max_rho = max_corr(corr_df, max_rho)
#     corr_rho = corr_df[['NDVI_SWI_001_rho', 'NDVI_SWI_010_rho',
#                         'NDVI_SWI_020_rho', 'NDVI_SWI_040_rho',
#                         'NDVI_SWI_060_rho', 'NDVI_SWI_100_rho']]
#     max_corr_val.append(corr_rho.max(axis=0).max())
#     max_corr_swi.append(corr_rho.max(axis=0).idxmax())
#     max_corr_lag.append(corr_rho.max(axis=1).idxmax())
#   
# max_corr_val = np.array(max_corr_val)
# max_corr_swi = np.array(max_corr_swi)
# max_corr_lag = np.array(max_corr_lag)
#   
# np.save('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_corr_val_0.npy', 
#         max_corr_val)
# np.save('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_corr_swi_0.npy', 
#         max_corr_swi)
# np.save('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_corr_lag_0.npy', 
#         max_corr_lag)
# 
#===========================================================================
#print max_rho
#np.save('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_rho_2007_2015.npy', 
#        max_rho)

#===========================================================================
# max_rho = np.load('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\max_rho_2007_2015.npy').item()
#     
# # read LC 300m
# #lc = read_LC(lcpath, 14.7, 29.4, 68, 81.8)
#     
# lccs_masked = LC_mask(lons, lats, search_rad=80000)
# #scatterplot(lons, lats,lccs_masked, s=75, title='ESA CCI land cover classes, 0.4 deg.')
#         
# max_rho_masked = {}
# for key in max_rho:
#     max_rho_masked[key] = np.ma.array(max_rho[key], mask=lccs_masked.mask)
#          
# # plot maps showing time lag with highest rho
# plot_rho(max_rho_masked, lons, lats)
#===========================================================================