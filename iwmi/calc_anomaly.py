from SWI_drought_indicator import get_lonlat_district
from simon import anomaly, read_ts_area, plot_anomaly
import matplotlib.pyplot as plt
from nc_stack_uptodate import check_stack
from readers import read_cfg
import ast
import os

cfg = read_cfg('config_file_10daily.cfg')

data_path = cfg['data_path']
data_path_nc = cfg['data_path_nc']
nc_stack_path = cfg['nc_stack_path']
variables = cfg['variables'].split()
swi_stack_name = cfg['swi_stack_name']
datestr = ast.literal_eval(cfg['datestr'])
t_value = 'SWI_' + cfg['t_value'].zfill(3)

check_stack(data_path, data_path_nc, nc_stack_path, swi_stack_name, variables, datestr)

input_path = os.path.join(nc_stack_path, swi_stack_name)
district = cfg['district']

# calculate latitude min/max and longitude min/max of wanted area
lat_min, lat_max, lon_min, lon_max = get_lonlat_district(district)
swi = read_ts_area(input_path, t_value, lat_min, lat_max, lon_min, lon_max)
swi_anom = anomaly(swi)
plot_anomaly(swi, swi_anom)
plt.show()
