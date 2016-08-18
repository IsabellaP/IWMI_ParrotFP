import os
from datetime import datetime
from poets.poet import Poet

# poets attributes:
rootpath = os.path.join('C:\\', 'Users', 'i.pfeil', 'Desktop', 'poets')
shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                         '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 'IND_adm1')
regions = ['Maharashtra'] # CE...Sri Lanka, IN...India
spatial_resolution = 0.4
temporal_resolution = 'monthly'
start_date = datetime(2007, 7, 1)
nan_value = -99

# initializing Poet class:
p = Poet(rootpath, regions, spatial_resolution, temporal_resolution,
         start_date, nan_value, shapefile=shapefile)

#===============================================================================
# #source attributes:
# name = 'NDVI'
# filename = "g2_BIOPAR_NDVI_QL_{YYYY}{MM}{DD}0000_ASIA_VGT_V1_3.tiff"
# filedate = {'YYYY': (18, 22), 'MM': (22, 24), 'DD': (24, 26)}
# temp_res = 'daily'
# host = "neoftp.sci.gsfc.nasa.gov"
# protocol = 'FTP'
# directory = "/gs/MOD11C1_D_LSTDA/"
# begin_date = datetime(2007, 1, 1)
# nan_value = 255
#    
# # initializing the data source:
# p.add_source(name, filename, filedate, temp_res, host, protocol,
#              begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r')
#===============================================================================
   
   
name = 'LAI'
filename = "g2_BIOPAR_LAI_QL_{YYYY}{MM}{DD}0000_VGT_V1.3.tiff"
filedate = {'YYYY': (17, 21), 'MM': (21, 23), 'DD': (23, 25)}
temp_res = 'daily'
host = "neoftp.sci.gsfc.nasa.gov"
protocol = 'FTP'
directory = "/gs/MOD11C1_D_LSTDA/"
begin_date = datetime(2007, 7, 1)
nan_value = 255
    
p.add_source(name, filename, filedate, temp_res, host, protocol,
             begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r')


#===============================================================================
# name = 'FAPAR'
# filename = "g2_BIOPAR_FAPAR_QL_{YYYY}{MM}{DD}0000_VGT_V1.3.tiff"
# filedate = {'YYYY': (19, 23), 'MM': (23, 25), 'DD': (25, 27)}
# temp_res = 'daily'
# host = "neoftp.sci.gsfc.nasa.gov"
# protocol = 'FTP'
# directory = "/gs/MOD11C1_D_LSTDA/"
# begin_date = datetime(2007, 1, 1)
# nan_value = 255
#     
# p.add_source(name, filename, filedate, temp_res, host, protocol,
#              begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r')
#===============================================================================

  
name = 'SWI'
filename = "g2_BIOPAR_SWI10_QL_200707110000_GLOBE_ASCAT_V3_0_1.tiff"
filedate = {'YYYY': (19, 23), 'MM': (23, 25), 'DD': (25, 27)}
temp_res = 'daily'
host = "neoftp.sci.gsfc.nasa.gov"
protocol = 'FTP'
directory = "/gs/MOD11C1_D_LSTDA/"
begin_date = datetime(2007, 7, 1)
nan_value = 255
  
p.add_source(name, filename, filedate, temp_res, host, protocol,
             begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r')
 
# for all sources:
begin = datetime(2007,7,1)
end = datetime(2010,12,31)
p.resample(begin=begin, end=end)

p.start_app()