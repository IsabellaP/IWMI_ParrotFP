from unzip import merge_nc, format_to_folder, merge_tiff, read_tiff
import shutil
from datetime import datetime
from netCDF4 import Dataset, num2date, date2num
import os
from osgeo import gdal, osr

def array_to_raster(array, lon, lat, dst_filename):

    # You need to get those values like you did.
    x_pixels = array.shape[1]  # number of pixels in x
    y_pixels = array.shape[0] # number of pixels in y
    PIXEL_SIZE = 0.1  # size of the pixel...
    x_min = lon.min() - PIXEL_SIZE/2
    y_max = lat.max() + PIXEL_SIZE/2 # x_min & y_max are like the "top left" corner.

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(dst_filename, x_pixels, y_pixels, 1, gdal.GDT_Float32, )

    dataset.SetGeoTransform((x_min, PIXEL_SIZE,  0, y_max, 0, -PIXEL_SIZE))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.GetRasterBand(1).SetNoDataValue(-99)
    dataset.FlushCache()  # Write to disk.
    return dataset, dataset.GetRasterBand(1)

# checks if stack exists, if False, stack will be created
# if stack already exists, it checks if the stack is up to date with the files in data_path
# if not up to date, the new files are appended to the existing stack
def check_stack(data_path, data_path_nc, nc_stack_path, swi_stack_name, variables, datestr):

    print "Check if data stack is up to date"
    if os.path.isfile(os.path.join(nc_stack_path, swi_stack_name)) == False:
        print "Create new netcdf stack of data"
        if not os.path.exists(data_path_nc):
            os.makedirs(data_path_nc)
        format_to_folder(data_path, '.nc', data_path_nc)
        merge_nc(data_path_nc, nc_stack_path, swi_stack_name, variables, datestr)
        print "Finished"
        shutil.rmtree(data_path_nc)
    else:
        nc_file = os.path.join(nc_stack_path, swi_stack_name)
        with Dataset(nc_file, 'a') as ncfile:
            unit_temps = ncfile.variables['time'].units
            nctime = ncfile.variables['time'][:]
            try:
                cal_temps = ncfile.variables['time'].calendar
            except AttributeError:  # Attribute doesn't exist
                cal_temps = u"gregorian"  # or standard

            nc_all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)
            nc_all_dates_str = [datetime.strftime(date, "%Y%m%d") for date in nc_all_dates]
            folder_all_dates = os.listdir(data_path)
            s = set(nc_all_dates_str)
            dates_to_append_str = [x for x in folder_all_dates if x not in s]
            dates_to_append = [datetime.strptime(date, "%Y%m%d") for date in dates_to_append_str]

            if dates_to_append:
                print 'Append new data to stack'
                for idx, date_folder in enumerate(dates_to_append_str):
                    path_to_nc = os.path.join(data_path, date_folder)
                    nc_date = os.listdir(path_to_nc)[0]
                    print date_folder
                    with Dataset(os.path.join(path_to_nc, nc_date), 'r') as ncfile_single:
                        data_single = {}
                        for var in variables:
                            data_single[var] = ncfile_single.variables[var][:]
                            ncfile[var][nc_all_dates.size+idx, :, :] = \
                                        data_single[var]

                    numdate = date2num(dates_to_append[idx], units=unit_temps, calendar=cal_temps)
                    ncfile.variables['time'][nc_all_dates.size+idx] = numdate
                print 'Finished - Stack is up to date again'
            else:
                print "Data stack is already up to date"
                
        
def check_tiff_stack(data_path, data_path_tif, stack_path, stack_name, variables, 
                     datestr):
    
    print "Check if data stack is up to date"
    if os.path.isfile(os.path.join(stack_path, stack_name)) == False:
        print "Create new tiff stack of data"
        if not os.path.exists(data_path_tif):
            os.makedirs(data_path_tif)
        format_to_folder(data_path, '.tif', data_path_tif)
        merge_tiff(data_path_tif, stack_path, stack_name, variables, datestr)
        print "Finished"
        shutil.rmtree(data_path_tif)
    else:
        nc_file = os.path.join(stack_path, stack_name)
        with Dataset(nc_file, 'a') as ncfile:
            unit_temps = ncfile.variables['time'].units
            nctime = ncfile.variables['time'][:]
            try:
                cal_temps = ncfile.variables['time'].calendar
            except AttributeError:  # Attribute doesn't exist
                cal_temps = u"gregorian"  # or standard

            nc_all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)
            nc_all_dates_str = [datetime.strftime(date, "%Y%m%d") for date in nc_all_dates]
            folder_all_dates = os.listdir(data_path)
            s = set(nc_all_dates_str)
            dates_to_append_str = [x for x in folder_all_dates if x not in s]
            dates_to_append = [datetime.strptime(date, "%Y%m%d") for date in dates_to_append_str]

            if dates_to_append:
                print 'Append new data to stack'
                for idx, date_folder in enumerate(dates_to_append_str):
                    path_to_tif = os.path.join(data_path, date_folder)
                    nc_date = os.listdir(path_to_tif)[0]
                    print date_folder
                    data_single, _, _ = read_tiff(path_to_tif, date_folder)
                    ncfile[variables][nc_all_dates.size+idx, :, :] = data_single

                    numdate = date2num(dates_to_append[idx], units=unit_temps, calendar=cal_temps)
                    ncfile.variables['time'][nc_all_dates.size+idx] = numdate
                print 'Finished - Stack is up to date again'
            else:
                print "Data stack is already up to date"