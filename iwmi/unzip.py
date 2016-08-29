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
                

def merge_nc(path, new_path, variables, datestr):
    
    new_ncf = 'new_nc.nc'
    
    # read one ncfile to get lon, lat settings
    with Dataset(os.path.join(path, os.listdir(path)[0]), 'r') as ncfile:
        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]
    
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
        data = {}
        for var in variables:
            data[var] = ncfile.createVariable(var, np.dtype('f4').char,
                                              ('time', 'lat', 'lon',),
                                              fill_value=255, zlib=True,
                                              complevel=4)

        for idx, ncf in enumerate(sorted(os.listdir(path))):
            print idx
            with Dataset(os.path.join(path, ncf), 'r') as ncfile_single:
                data_single = {}
                for var in variables:
                    data_single[var] = ncfile_single.variables[var][:]
                    data[var][idx,:,] = data_single[var]

            year = int(ncf[datestr['year'][0]:datestr['year'][1]])
            month = int(ncf[datestr['month'][0]:datestr['month'][1]])
            day = int(ncf[datestr['day'][0]:datestr['day'][1]])
            numdate = date2num(datetime(year,month,day), units=times.units,
                               calendar=times.calendar)
            times[idx] = numdate     
        
    print 'Finished.'

if __name__ == '__main__':
    
    #path_in = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\FAPAR_zipped'
    #path_out = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\FAPAR'
    
    #unzip(path_in, path_out)
    
    root = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\VIs\\LAI\\'
    formatstr = '.nc'
    out_path = 'C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\FAPAR\\'
    #format_to_folder(root, formatstr, out_path)
    
    new_path = 'C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\'
    #datestr = {'year': (16,20), 'month': (20,22), 'day': (22,24)} # SWI
    #datestr = {'year': (15,19), 'month': (19,21), 'day': (21,23)} # NDVI
    datestr = {'year': (16,20), 'month': (20,22), 'day': (22,24)} # FAPAR
    #datestr = {'year': (14,18), 'month': (18,20), 'day': (20,22)} # LAI
    merge_nc(out_path, new_path, ['FAPAR'], datestr)
    