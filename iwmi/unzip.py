import os
import numpy as np
import shutil
import zipfile
from datetime import datetime, timedelta
from netCDF4 import Dataset, date2num
import gdal
from zipfile import BadZipfile


def find_nearest(array, element):
    idx = (np.abs(array - element)).argmin()
    return array[idx], idx


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
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    for path, _, files in os.walk(root):
        for name in files:
            if formatstr in name:
                shutil.copyfile(os.path.join(path, name), 
                                os.path.join(out_path, name))
                

def merge_nc(path, new_path, new_ncf, variables, datestr, lat_min=None,
             lat_max=None, lon_min=None, lon_max=None, formatfolder=True):

    # read one ncfile to get lon, lat settings
    if formatfolder:
        with Dataset(os.path.join(path, os.listdir(path)[0]), 'r') as ncfile:
            lons = ncfile.variables['lon'][:]
            lats = ncfile.variables['lat'][:]
    else:
        count = 0
        formatstr = '.nc'
        for subpath, _, files in os.walk(path):
            for name in files:
                if formatstr in name:
                    if count > 0:
                        break
                    with Dataset(os.path.join(subpath, os.listdir(subpath)[0]), 'r') as ncfile:
                        lons = ncfile.variables['lon'][:]
                        lats = ncfile.variables['lat'][:]
                    count += 1
                    
    
    if lat_min and lat_max and lon_min and lon_max:
        lon_min, idx1 = find_nearest(lons, lon_min)
        lon_max, idx2 = find_nearest(lons, lon_max)
        lat_min, idx3 = find_nearest(lats, lat_min)
        lat_max, idx4 = find_nearest(lats, lat_max)
        lat_size = np.abs(idx3-idx4)
        lon_size = np.abs(idx1-idx2)
    
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
        if lat_min and lat_max and lon_min and lon_max:
            arrlat = lats[idx4:idx3]
            arrlon = lons[idx1:idx2]
        else:
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

        if formatfolder:
            for idx, ncf in enumerate(sorted(os.listdir(path))):
                print idx, ncf
                with Dataset(os.path.join(path, ncf), 'r') as ncfile_single:
                    data_single = {}
                    for var in variables:
                        if lat_min and lat_max and lon_min and lon_max:
                            data_single[var] = ncfile_single.variables[var][0, idx4:idx3, idx1:idx2]
                        else:
                            data_single[var] = ncfile_single.variables[var][:]
                        data[var][idx,:,:] = data_single[var]

            year = int(ncf[datestr['year'][0]:datestr['year'][1]])
            month = int(ncf[datestr['month'][0]:datestr['month'][1]])
            day = int(ncf[datestr['day'][0]:datestr['day'][1]])
            numdate = date2num(datetime(year,month,day), units=times.units,
                               calendar=times.calendar)
            times[idx] = numdate     

        else:
            idx = 0
            for subpath, dirs, files in os.walk(path):
                dirs.sort()
                for ncf in sorted(files):
                    if formatstr in ncf:
                        print idx, ncf
                        with Dataset(os.path.join(subpath, ncf), 'r') as ncfile_single:
                            data_single = {}
                            for var in variables:
                                if lat_min and lat_max and lon_min and lon_max:
                                    data_single[var] = ncfile_single.variables[var][idx4:idx3, idx1:idx2]
                                else:
                                    data_single[var] = ncfile_single.variables[var][:]
                                data[var][idx,:,:] = data_single[var]

                            year = int(ncf[datestr['year'][0]:datestr['year'][1]])
                            month = int(ncf[datestr['month'][0]:datestr['month'][1]])
                            day = int(ncf[datestr['day'][0]:datestr['day'][1]])
                            numdate = date2num(datetime(year,month,day), units=times.units,
                                               calendar=times.calendar)
                            times[idx] = numdate
                            
                        idx += 1
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




if __name__ == '__main__':
       
    path = "/data/Copernicus/LAI300/all/"
    #path_uz = "/data/Copernicus/SWI/Kenya_or_Eth2/uz"
    #unzip(path, path_uz)

    new_path = "/data/Copernicus/LAI300/stack/"
    #format_path = "/data/Copernicus/LAI300/nc/"
    #format_to_folder(path, '.nc', format_path)

    country = 'Austria'
    new_ncf = 'LAI300_stack_AT_2.nc'
    #variables = ['SWI_001', 'SWI_005', 'SWI_010', 'SWI_015',
    #             'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']
    variables = ['LAI','QFLAG']#, 'TIME_GRID', 'crs']
    #datestr = {'year': [16,20], 'month': [20,22], 'day': [22,24]} #SWI
    datestr = {'year': [13,17], 'month': [17,19], 'day': [19,21]} #LAI
 
    if country == 'Kenya':
        latlon = [-4.6696, 4.6224, 33.9072, 41.9051]
    elif country == 'DRC':
        latlon = [-13.4580, 5.3806, 12.2145, 31.3027]
    if country == 'Ethiopia':
        latlon = [3.4066, 14.8836, 32.9917, 47.9882]
    if country == 'Colombia':
        latlon = [-4.2368, 12.5902, -81.7201, -66.8704]
        #latlon=[3.087138, 3.795309, -76.735399, -76.019913]
    if country == 'Austria':
        latlon = [46.4074, 49.0187, 9.5335, 17.1663]
    if country == 'India':
        latlon = [6.7458, 35.5056, 68.1442, 97.3805]
    print new_ncf
    lat_min=latlon[0]
    lat_max=latlon[1]
    lon_min=latlon[2]
    lon_max=latlon[3]
    # formatfolder false should work as long as data is not zipped
    merge_nc(path, new_path, new_ncf, variables, datestr,
             lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
             formatfolder=False)
    
    print "done"