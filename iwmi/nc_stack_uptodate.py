import os
import shutil
import ast
from osgeo import gdal, osr
from datetime import datetime, timedelta
from netCDF4 import Dataset, num2date, date2num
from veg_pred_preprocessing import merge_nc, format_to_folder, merge_tiff, read_tiff, unzip, read_cfg


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

    print "Check if nc data stack is up to date"
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
                            ncfile[var][nc_all_dates.size+idx, 
                                        :data_single[var].shape[0], :] = \
                                        data_single[var]

                    numdate = date2num(dates_to_append[idx], units=unit_temps, calendar=cal_temps)
                    ncfile.variables['time'][nc_all_dates.size+idx] = numdate
                print 'Finished - Stack is up to date again'
            else:
                print "Data stack is already up to date"
                
        
def check_tiff_stack(data_path, data_path_tif, stack_path, stack_name, variables, 
                     datestr):
    
    print "Check if tif data stack is up to date"
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
            
            for fname in folder_all_dates:
                if fname[9] == '_':
                    year = fname[5:9]
                    doy = fname[10:13]
                    date = datetime(int(year), 1, 1) + timedelta(int(doy) - 1)
                    f_new = 'NDVI_'+str(year)+str(date.month).zfill(2)+str(date.day).zfill(2)+'.tif'
                    os.rename(os.path.join(data_path, fname), os.path.join(data_path, f_new))
            
            folder_all_dates = os.listdir(data_path)
            folder_all_dates_str = [date[datestr['year'][0]:datestr['day'][1]] for date in folder_all_dates]
            s = set(nc_all_dates_str)
            dates_to_append_str = [x for x in folder_all_dates_str if x not in s]
            dates_to_append = [datetime.strptime(date, "%Y%m%d") for date in dates_to_append_str]

            if dates_to_append:
                print 'Append new data to stack'
                for idx, date_str in enumerate(dates_to_append_str):
                    tif_name = 'NDVI_'+date_str+'.tif'
                    print tif_name
                    data_single = read_tiff(data_path, tif_name, lonlat=False)
                    ncfile[variables][nc_all_dates.size+idx, :, :] = data_single

                    numdate = date2num(dates_to_append[idx], units=unit_temps, calendar=cal_temps)
                    ncfile.variables['time'][nc_all_dates.size+idx] = numdate
                print 'Finished - Stack is up to date again'
            else:
                print "Data stack is already up to date"

if __name__ == '__main__':
    # check and update SWI stack
    cfg = read_cfg('config_file_daily.cfg')
     
    swi_zippath = cfg['swi_zippath']
    data_path = cfg['swi_rawdata']
    unzip(swi_zippath, data_path)
      
    data_path_nc = cfg['swi_path_nc']
    nc_stack_path = cfg['swi_path']
    swi_stack_name = cfg['swi_stack_name']
    variables = cfg['swi_variables'].split()
    datestr = ast.literal_eval(cfg['swi_datestr'])
      
    check_stack(data_path, data_path_nc, nc_stack_path, swi_stack_name, 
                variables, datestr)
     
    # check and update VI stack
    data_path = cfg['vi_rawdata']
    data_path_nc = cfg['vi_path_nc']
    nc_stack_path = cfg['vi_path']
    swi_stack_name = cfg['vi_stack_name']
    variables = cfg['vi_variables']
    datestr = ast.literal_eval(cfg['vi_datestr'])
     
    check_tiff_stack(data_path, data_path_nc, nc_stack_path, swi_stack_name, 
                     variables, datestr)
