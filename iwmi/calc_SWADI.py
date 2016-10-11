from SWI_drought_indicator import get_lonlat_district, create_SWADI_dist, read_mask, discrete_cmap, drought_index
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import matplotlib.colors as cls
import os
from nc_stack_uptodate import check_stack, array_to_raster
from readers import read_cfg
import ast
import numpy as np
from datetime import datetime

cfg = read_cfg('config_file_10daily.cfg')

data_path = cfg['data_path']
data_path_nc = cfg['data_path_nc']
nc_stack_path = cfg['nc_stack_path']
variables = cfg['variables'].split()
swi_stack_name = cfg['swi_stack_name']
datestr = ast.literal_eval(cfg['datestr'])
t_value = 'SWI_' + cfg['t_value'].zfill(3)

stack_input_path = os.path.join(nc_stack_path, swi_stack_name)
district = cfg['district']
day_from = cfg['day_from']
day_to = cfg['day_to']
start_date = datetime.strptime(day_from, '%Y-%m-%d').date()
end_date = datetime.strptime(day_to, '%Y-%m-%d').date()
out_path = cfg['out_path']

check_stack(data_path, data_path_nc, nc_stack_path, swi_stack_name, variables, datestr)

# calculate latitude min/max and longitude min/max of wanted area
lat_min, lat_max, lon_min, lon_max = get_lonlat_district(district)

# calculate SWADI
SWADI, anomalies, lon, lat = drought_index(stack_input_path, t_value, lat_min, lat_max, lon_min, lon_max)

# Get days
all_days = SWADI.index.date
date_idx = np.where((all_days >= start_date) & (all_days <= end_date))[0]
days = [(all_days[i]).strftime('%Y-%m-%d') for i in date_idx]

# lon/lat values and size for GTiff
lon_u = np.unique(lon)
lat_u = np.unique(lat)
x_size = lon_u.size
y_size = lat_u.size

# Calculate SWADI distribution for TS representation
SWADI_dist = create_SWADI_dist(stack_input_path, lat_min, lat_max, lon_min, lon_max, df=SWADI)

# Plot Time Series of SWADI
ax = SWADI_dist.plot.area(figsize=[30, 5], alpha=0.4, color='gyr')
ax.set_title('SWADI', fontsize=18)
plt.grid()
plt.axhline(0, color='black')
plt.ylim([-1, 1])
ax.set_xticks(SWADI_dist.index[::18])
ax.set_xticklabels(SWADI_dist.index[::18].date, fontsize=12)
ts_filename = os.path.join(out_path, 'SWADI_TS_' + district[-2:] + '.png')
plt.savefig(ts_filename, dpi=250, bbox_inches='tight', pad_inches=0.3)

# Writing selected dates to GTiff
out_path_tiff = os.path.join(out_path, 'SWADI')
out_path_anom = os.path.join(out_path, 'Anomalies')
if not os.path.exists(out_path_tiff):
    os.makedirs(out_path_tiff)
if not os.path.exists(out_path_anom):
    os.makedirs(out_path_anom)
for day in days:
    # SWADI
    filename_path_SWADI = os.path.join(out_path_tiff, day.replace('-', '_') + '.tiff')
    arr = SWADI.loc[day].values
    arr = np.reshape(map(int, arr), [y_size, x_size])
    array_to_raster(arr, lon_u, lat_u, filename_path_SWADI)

    # Anomalies
    filename_path_anom = os.path.join(out_path_anom, day.replace('-', '_') + '.tiff')
    arr = anomalies.loc[day].values
    arr = np.reshape(map(int, arr), [y_size, x_size])
    array_to_raster(arr, lon_u, lat_u, filename_path_anom)

# if days is not None:
#     for day in days:
#         lc_mask = read_mask(lat_min, lat_max, lon_min, lon_max)
#         mask = lc_mask.data.flatten()
#         mask[np.where(mask != 1)] = 0
#         mask_ind = np.where(mask == 1)
#
#         date = pd.to_datetime(day).strftime('%Y-%m-%d')
#         fig = plt.figure()
#         map = Basemap(projection='cyl', llcrnrlon=lon_min-0.5, llcrnrlat=lat_min-0.5, urcrnrlat=lat_max+0.5,
#                       urcrnrlon=lon_max+0.5)
#         map.drawmapboundary()
#         map.drawcountries()
#         cmap = cls.ListedColormap(['#A20000', '#DE0000', '#E5E618', '#47A917', '#30740C'], name='from_list', N=None)
#         map.scatter(np.array(lon).flatten()[mask_ind], np.array(lat).flatten()[mask_ind],
#                     c=SWADI.loc[date].values.flatten()[mask_ind], edgecolor='None',
#                     marker='s', s=pixelsize, vmin=1, vmax=5, cmap=discrete_cmap(6, cmap))
#         map.readshapefile(shapefile, district, linewidth=0.4)
#         cbar = plt.colorbar(ticks=range(6))
#         cbar.ax.tick_params(labelsize=18)
#
#         plt.clim(0.5, 6 - 0.5)
#         plt.title('SWADI at ' + date, fontsize=22)
#
# plt.show()
pass