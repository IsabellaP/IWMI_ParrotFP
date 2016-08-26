import os
import numpy as np
import shutil
import zipfile
from datetime import datetime
from netCDF4 import Dataset, date2num


def unzip(path_in, path_out):
    """ Unzips folders from path_in to path_out 
    """
        
    folders = os.listdir(path_in)
    
    for fname in folders:
        zipfile_n = os.listdir(os.path.join(path_in, fname))[1]
        zip_ref = zipfile.ZipFile(os.path.join(path_in, fname, zipfile_n), 'r')
        zip_ref.extractall(path_out)
        zip_ref.close()
        
        
def format_to_folder(root, formatstr, out_path):
    """ Looks for files of format formatstr in all subfolders of root and moves
    the files to out_path
    """
    
    for path, _, files in os.walk(root):
        for name in files:
            if formatstr in name:
                shutil.copyfile(os.path.join(path, name), 
                                os.path.join(out_path, name))
                

def merge_nc(path, new_path):
    
    new_ncf = 'new_nc.nc'
    
    # read one ncfile to get lon, lat settings
    with Dataset(os.path.join(path, os.listdir(path)[0]), 'r') as ncfile:
        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]
        
    swi1_tmp = []
    swi2_tmp = []
    swi3_tmp = []
    swi4_tmp = []
    swi5_tmp = []
    swi6_tmp = []
    swi7_tmp = []
    swi8_tmp = []
    date = []
    
    with Dataset(os.path.join(new_path, new_ncf), 'w') as ncfile:
        
        print 'Writing data to netCDF...'

        start_date = datetime(year=2000, month=1, day=1)

        ncfile.createDimension('time', None)
        times = ncfile.createVariable('time', 'uint16', ('time',),
                                      zlib=True, complevel=4)
        setattr(times, 'long_name', 'Time')
        times.units = 'days since ' + str(start_date)
        times.calendar = 'standard'

        # Define lat and lon as dim and var
        arrlat = lats
        arrlon = lons
        latsize = len(arrlat)
        lonsize = len(arrlon)

        ncfile.createDimension('lat', latsize)
        ncfile.createDimension('lon', lonsize)

        lat = ncfile.createVariable('lat', np.dtype('f4').char,
                                    ('lat',),
                                    zlib=True, complevel=4)

        lat[:] = arrlat
        setattr(lat, 'long_name', 'Latitude')
        setattr(lat, 'units', 'degrees_north')
        setattr(lat, 'standard_name', 'latitude')
        setattr(lat, 'valid_range', [-90.0, 90.0])

        lon = ncfile.createVariable('lon', np.dtype('f4').char,
                                    ('lon',),
                                    zlib=True, complevel=4)

        lon[:] = arrlon
        setattr(lon, 'long_name', 'Longitude')
        setattr(lon, 'units', 'degrees_east')
        setattr(lon, 'standard_name', 'longitude')
        setattr(lon, 'valid_range', [-180.0, 180.0])

        # variables
        swi1 = ncfile.createVariable('SWI_001', np.dtype('f4').char,
                                    ('time', 'lat', 'lon',),
                                    fill_value=255, zlib=True,
                                    complevel=4)
        
        swi2 = ncfile.createVariable('SWI_005', np.dtype('f4').char,
                                    ('time', 'lat', 'lon',),
                                    fill_value=255, zlib=True,
                                    complevel=4)
        
        swi3 = ncfile.createVariable('SWI_010', np.dtype('f4').char,
                                    ('time', 'lat', 'lon',),
                                    fill_value=255, zlib=True,
                                    complevel=4)
                    
        swi4 = ncfile.createVariable('SWI_015', np.dtype('f4').char,
                                    ('time', 'lat', 'lon',),
                                    fill_value=255, zlib=True,
                                    complevel=4)
        
        swi5 = ncfile.createVariable('SWI_020', np.dtype('f4').char,
                                    ('time', 'lat', 'lon',),
                                    fill_value=255, zlib=True,
                                    complevel=4)
        
        swi6 = ncfile.createVariable('SWI_040', np.dtype('f4').char,
                                    ('time', 'lat', 'lon',),
                                    fill_value=255, zlib=True,
                                    complevel=4)
        
        swi7 = ncfile.createVariable('SWI_060', np.dtype('f4').char,
                                    ('time', 'lat', 'lon',),
                                    fill_value=255, zlib=True,
                                    complevel=4)
        
        swi8 = ncfile.createVariable('SWI_100', np.dtype('f4').char,
                                    ('time', 'lat', 'lon',),
                                    fill_value=255, zlib=True,
                                    complevel=4)

        for ncf in sorted(os.listdir(path)):  
            with Dataset(os.path.join(path, ncf), 'r') as ncfile_single:
                swi1_single = ncfile_single.variables['SWI_001'][:]
                swi2_single = ncfile_single.variables['SWI_005'][:]
                swi3_single = ncfile_single.variables['SWI_010'][:]
                swi4_single = ncfile_single.variables['SWI_015'][:]
                swi5_single = ncfile_single.variables['SWI_020'][:]
                swi6_single = ncfile_single.variables['SWI_040'][:]
                swi7_single = ncfile_single.variables['SWI_060'][:]
                swi8_single = ncfile_single.variables['SWI_100'][:]
                
            swi1_tmp.append(swi1_single)
            swi2_tmp.append(swi2_single)
            swi3_tmp.append(swi3_single)
            swi4_tmp.append(swi4_single)
            swi5_tmp.append(swi5_single)
            swi6_tmp.append(swi6_single)
            swi7_tmp.append(swi7_single)
            swi8_tmp.append(swi8_single)
            
            year = int(ncf[16:20])
            month = int(ncf[20:22])
            day = int(ncf[22:24])
            date.append(datetime(year, month, day))
                
        timestamp = date
        numdate = date2num(timestamp, units=times.units,
                       calendar=times.calendar)
        times[:] = numdate
        
        swi1[:] = swi1_tmp
        swi2[:] = swi2_tmp
        swi3[:] = swi3_tmp
        swi4[:] = swi4_tmp
        swi5[:] = swi5_tmp
        swi6[:] = swi6_tmp
        swi7[:] = swi7_tmp
        swi8[:] = swi8_tmp

if __name__ == '__main__':
    
    #path_in = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\FAPAR_zipped'
    #path_out = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\FAPAR'
    
    #unzip(path_in, path_out)
    
    root = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\SWI'
    formatstr = '.nc'
    out_path = 'C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\SWI'
    #format_to_folder(root, formatstr, out_path)
    new_path = 'C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\'
    merge_nc(out_path, new_path)
    