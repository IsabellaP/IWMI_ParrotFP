import os
from simon import read_ts_area
from data_analysis import rescale_peng
import matplotlib.pyplot as plt
from poets.shape.shapes import Shape


def swi_ndvi_region(path, param, lat_min, lat_max, lon_min, lon_max, t=40):
    df_swi = read_ts_area(path, param, lat_min, lat_max, lon_min, lon_max, t=t)
    
    vi_path = "C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA2\\NDVI\\NDVI_stack.nc"
    vi_param = 'NDVI'
    df_vi = read_ts_area(vi_path, vi_param, lat_min, lat_max, lon_min, lon_max)
    
    swi_resc = rescale_peng(df_swi, df_swi.min(), df_swi.max())
    vi_resc = rescale_peng(df_vi, df_vi.min(), df_vi.max())
    
    ax=swi_resc.plot()
    vi_resc.plot(ax=ax)
    plt.title('lat min, max: '+str(lat_min)+', '+str(lat_max)+', '+
              'lon min, max: '+str(lon_min)+', '+str(lon_max))
    plt.show()


if __name__ == '__main__':
    
    path = 'C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA2\\SWI\\SWI_stack.nc'
    param = 'SWI'
    
    shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                             '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 'IND_adm2')
    region = 'IN.MH.JN'
    
    shpfile = Shape(region, shapefile=shapefile)
    lon_min, lat_min, lon_max, lat_max = shpfile.bbox
    
    swi_ndvi_region(path, param, lat_min, lat_max, lon_min, lon_max)