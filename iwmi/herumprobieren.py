from simon import read_ts_area
from data_analysis import rescale_peng
import matplotlib.pyplot as plt

path = 'C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\SWI\\SWI_stack.nc'
param = 'SWI'

lat_min = 21.204
lat_max = 23
lon_min = 75
lon_max = 77.5754
df_10 = read_ts_area(path, param, lat_min, lat_max, lon_min, lon_max, t=40)
#df_20 = read_ts_area(path, param, lat_min, lat_max, lon_min, lon_max, t=20)
#df_60 = read_ts_area(path, param, lat_min, lat_max, lon_min, lon_max, t=60)

vi_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\NDVI\\NDVI_stack.nc"
vi_param = 'NDVI'
df_vi = read_ts_area(vi_path, vi_param, lat_min, lat_max, lon_min, lon_max)

swi10_resc = rescale_peng(df_10, df_10.min(), df_10.max())
#swi20_resc = rescale_peng(df_20, df_20.min(), df_20.max())
#swi60_resc = rescale_peng(df_60, df_60.min(), df_60.max())
vi_resc = rescale_peng(df_vi, df_vi.min(), df_vi.max())

ax=swi10_resc.plot()
#swi20_resc.plot(ax=ax)
#swi60_resc.plot(ax=ax)
vi_resc.plot(ax=ax)
plt.title('lat min, max: '+str(lat_min)+', '+str(lat_max)+', '+
          'lon min, max: '+str(lon_min)+', '+str(lon_max))
plt.show()

print "done"
