import os
from datetime import datetime
from poets.poet import Poet

#poets attributes:
#rootpath = 'C:\Users\s.hochstoger\Desktop\poets'
#shapefile = os.path.join('C:\\', 'Users', 's.hochstoger', 'Desktop',
#                         'poets', 'Box_West_SA', 'West_SA_cl2')
rootpath = os.path.join('C:\\', 'Users', 'i.pfeil', 'Desktop', 'poets')

#rootpath = "E:\\poets\\"
#shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
#                         '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 'IND_adm1')
shapefile = os.path.join('C:\\', 'Users', 'i.pfeil', 'Desktop', 
                         'Isabella', 'Peejush', 'Box_West_SA', 'West_SA_cl2')
regions = ['West_SA'] # CE...Sri Lanka, IN...India
spatial_resolution = 0.25
temporal_resolution = 'daily'
start_date = datetime(2001, 1, 1)
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

#
# name = 'LAI'
# filename = "g2_BIOPAR_LAI_QL_{YYYY}{MM}{DD}0000_VGT_V1.3.tiff"
# filedate = {'YYYY': (17, 21), 'MM': (21, 23), 'DD': (23, 25)}
# temp_res = 'daily'
# host = "neoftp.sci.gsfc.nasa.gov"
# protocol = 'FTP'
# directory = "/gs/MOD11C1_D_LSTDA/"
# begin_date = datetime(2007, 7, 1)
# nan_value = 255
#
# p.add_source(name, filename, filedate, temp_res, host, protocol,
#              begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r')

#===============================================================================
# name = 'VCI'
# filename = "VCI_{YYYY}_{MM}_{DD}.tif"
# filedate = {'YYYY': (4, 8), 'MM': (9, 11), 'DD': (12, 14)}
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
#
#
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
#
#
# name = 'SWI'
# filename = "g2_BIOPAR_SWI10_QL_200707110000_GLOBE_ASCAT_V3_0_1.tiff"
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
  
#===============================================================================
# name = 'SWI'
# filename = "g2_BIOPAR_SWI10_200707110000_GLOBE_ASCAT_V3_0_1.nc"
# filedate = {'YYYY': (16, 20), 'MM': (20, 22), 'DD': (22, 24)}
# temp_res = 'daily'
# host = "neoftp.sci.gsfc.nasa.gov"
# protocol = 'FTP'
# directory = "/gs/MOD11C1_D_LSTDA/"
# begin_date = datetime(2007, 7, 1)
# nan_value = 255
#     
# p.add_source(name, filename, filedate, temp_res, host, protocol,
#              begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r',
#              variables=['SWI_001', 'SWI_010', 'SWI_020', 'SWI_040', 'SWI_060', 
#                         'SWI_100'])
#===============================================================================

#===============================================================================
# name = 'LC'
# filename = "ESACCI-LC-L4-LCCS-Map-300m-P5Y-20100101-West_SA-v1.6.1.nc"
# filedate = {'YYYY': (31, 35), 'MM': (35, 37), 'DD': (37, 39)}
# temp_res = 'daily'
# host = "neoftp.sci.gsfc.nasa.gov"
# protocol = 'FTP'
# directory = "/gs/MOD11C1_D_LSTDA/"
# begin_date = datetime(2010, 1, 1)
# nan_value = 255
#         
# # initializing the data source:
# p.add_source(name, filename, filedate, temp_res, host, protocol,
#              begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r',
#              variables=['lccs_class'])
#===============================================================================

#===============================================================================
# name = 'IDSI'
# filename = "IDSI_{YYYY}_{MM}_{DD}.tif"
# filedate = {'YYYY': (5, 9), 'MM': (10, 12), 'DD': (13, 15)}
# temp_res = 'daily'
# host = "neoftp.sci.gsfc.nasa.gov"
# protocol = 'FTP'
# directory = "/gs/MOD11C1_D_LSTDA/"
# begin_date = datetime(2010, 1, 1)
# nan_value = 255
# 
# p.add_source(name, filename, filedate, temp_res, host, protocol,
#              begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r')
#===============================================================================

#===============================================================================
# # gapfree NDVI
# name = 'NDVI'
# filename = "NDVI_{YYYY}{MM}{DD}.nc"
# filedate = {'YYYY': (5, 9), 'MM': (9, 11), 'DD': (11, 13)}
# temp_res = 'daily'
# host = "neoftp.sci.gsfc.nasa.gov"
# protocol = 'FTP'
# directory = "/gs/MOD11C1_D_LSTDA/"
# begin_date = datetime(2001, 1, 1)
# nan_value = 255
#   
# # initializing the data source:
# p.add_source(name, filename, filedate, temp_res, host, protocol,
#              begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r')
#===============================================================================

# agriculture mask
name = 'AG_LC'
filename = "{YYYY}{MM}{DD}_AG_Mask.tif"
filedate = {'YYYY': (0, 4), 'MM': (4, 6), 'DD': (6, 8)}
temp_res = 'daily'
host = "neoftp.sci.gsfc.nasa.gov"
protocol = 'FTP'
directory = "/gs/MOD11C1_D_LSTDA/"
begin_date = datetime(2001, 1, 1)
nan_value = 0
 
# initializing the data source:
p.add_source(name, filename, filedate, temp_res, host, protocol,
             begin_date=begin_date, nan_value=nan_value, colorbar='terrain_r')

# for all sources:
begin = datetime(2001,1,1)
end = datetime(2001,1,7)
p.resample(begin=begin, end=end)
