import os
import numpy as np
import shutil
import zipfile
from datetime import datetime, timedelta
from netCDF4 import Dataset, date2num
import gdal
from zipfile import BadZipfile


def unzip(path_in, path_out):
    """ Unzips folders from path_in to path_out 
    """
        
    folders = os.listdir(path_in)
    
    for fname in folders:
        zipfile_n = os.listdir(os.path.join(path_in, fname))[1]
        try:
            zip_ref = zipfile.ZipFile(os.path.join(path_in, fname, zipfile_n), 'r')
        except BadZipfile:
            continue
        zip_ref.extractall(path_out)
        zip_ref.close()
        
        
def format_to_folder(root, formatstr, out_path):
    """ Looks for files of format formatstr in all subfolders of root and copies
    the files to out_path
    """
    
    for path, _, files in os.walk(root):
        for name in files:
            if formatstr in name:
                shutil.copyfile(os.path.join(path, name), 
                                os.path.join(out_path, name))
                

def merge_nc(path, new_path, new_ncf, variables, datestr):

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
                    if idx >= 1461:
                        data_single[var] = ncfile_single.variables[var][:-1,:]
                    else:
                        data_single[var] = ncfile_single.variables[var][:]
                    data[var][idx,:,:] = data_single[var]

            year = int(ncf[datestr['year'][0]:datestr['year'][1]])
            month = int(ncf[datestr['month'][0]:datestr['month'][1]])
            day = int(ncf[datestr['day'][0]:datestr['day'][1]])
            numdate = date2num(datetime(year,month,day), units=times.units,
                               calendar=times.calendar)
            times[idx] = numdate     
        
    print 'Finished.'
    
    
def read_tiff(srcpath, fname, coord_alt=False, lonlat=False):
    
    src_filename = os.path.join(srcpath, fname)

    # Opens source dataset
    src_ds = gdal.Open(src_filename)
    band = src_ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    latsize, lonsize = arr.shape
    
    if lonlat:
        if not coord_alt: # falsche coord!
            # unravel GDAL affine transform parameters
            c, a, b, f, d, e = src_ds.GetGeoTransform()
            
            col = np.arange(lonsize)
            row = np.arange(latsize)
            col_grid, row_grid = np.meshgrid(col, row)
            
            lons = a * col_grid + b * row_grid + a * 0.5 + b * 0.5 + c
            lats = d * col_grid + e * row_grid + d * 0.5 + e * 0.5 + f
        else:
            width = src_ds.RasterXSize
            height = src_ds.RasterYSize
            gt = src_ds.GetGeoTransform()
            minx = gt[0]
            miny = gt[3] + width*gt[4] + height*gt[5] 
            maxx = gt[0] + width*gt[1] + height*gt[2]
            maxy = gt[3]
            
            px = (maxx-minx)/lonsize
            py = (maxy-miny)/latsize
            
            lons = np.arange(minx, maxx, px)
            lats = np.arange(miny, maxy, py)
        
        return arr, lons, lats
    else:
        return arr


def merge_tiff(path, new_path, new_tiff, variable, datestr):
    
    # read one ncfile to get lon, lat settings
    _, lons, lats = read_tiff(path, os.listdir(path)[0])
    
    with Dataset(os.path.join(new_path, new_tiff), 'w') as ncfile:
        
        print 'Writing data to netCDF...'

        start_date = datetime(year=2000, month=1, day=1)

        ncfile.createDimension('time', None)
        times = ncfile.createVariable('time', 'uint16', ('time',),
                                      zlib=True, complevel=4)
        setattr(times, 'long_name', 'Time')
        times.units = 'days since ' + str(start_date)
        times.calendar = 'standard'

        # Define lat and lon as dim and var
        arrlat = lats[:,0]
        arrlon = lons[0,:]
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
        data = ncfile.createVariable(variable, np.dtype('f4').char,
                                          ('time', 'lat', 'lon',),
                                          fill_value=255, zlib=True,
                                          complevel=4)

        for idx, tiff in enumerate(sorted(os.listdir(path))):
            print idx
            data_single, _, _ = read_tiff(path, tiff)
            print data_single.shape
            if data_single.shape == lons.shape:
                data[idx,:,:] = data_single
            elif data_single.shape == (3518,7401):
                # hard coded for NDVI gapfree
                print idx, "dims change"
                data[idx,:-1,:] = data_single
            else:
                print "continue"
                continue
            
            year = int(tiff[datestr['year'][0]:datestr['year'][1]])
            month = int(tiff[datestr['month'][0]:datestr['month'][1]])
            day = int(tiff[datestr['day'][0]:datestr['day'][1]])
            numdate = date2num(datetime(year,month,day), units=times.units,
                               calendar=times.calendar)
            times[idx] = numdate     
        
    print 'Finished.'


def rename_files():
    
    path = 'E:\\_DATA\\NDVI_2001-15_gapfree\\WGS84\\'
    fnames = os.listdir(path)
    
    for f in fnames:
        print f, datetime.now()
        year = f[5:9]
        doy = f[10:13]
        date = datetime(int(year), 1, 1) + timedelta(int(doy) - 1)
        f_new = 'NDVI_'+str(year)+str(date.month).zfill(2)+str(date.day).zfill(2)+'.tif'
        os.rename(os.path.join(path, f), os.path.join(path, f_new))


if __name__ == '__main__':
       
    path = "E:\\_DATA\\SWI_daily\\"
    path_uz = "E:\\_DATA\\SWI_daily_unzipped\\"
    #unzip(path, path_uz)
    
    new_path = "E:\\_DATA\\"
    format_path = "E:\\_DATA\\SWI_daily_nc\\"
    #format_to_folder(path_uz, '.nc', format_path)
        
    new_ncf = 'SWI_daily_stack.nc'
    variables = ['SWI_001', 'SWI_005', 'SWI_010', 'SWI_015', 
                 'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']  
    datestr = {'year': [14,18], 'month': [18,20], 'day': [20,22]}
      
    merge_nc(format_path, new_path, new_ncf, variables, datestr)
    
    print "done"