import gdal
import datetime

years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]

def change_date_str(year, jd):
    doy = jd*8-7
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    date = date.strftime("%Y_%m_%d")
    return date

for year in years:
    src_filename = 'H:\\MODIS_VCI\\vci_' + str(year) + '.tif'

    # Opens source dataset
    src_ds = gdal.Open(src_filename)
    format = "GTiff"
    driver = gdal.GetDriverByName(format)

    proj = src_ds.GetProjection()
    geotransform = src_ds.GetGeoTransform()

    for i in range(src_ds.RasterCount):
        i += 1
        date = change_date_str(year, i)
        print date
        dst_filename = 'H:\\MODIS_VCI\\GTiffs\\VCI_' + date + '.tif'
        band = src_ds.GetRasterBand(i)
        nodatavalue = band.GetNoDataValue()
        arr = band.ReadAsArray()
        nx = band.XSize
        ny = band.YSize

        # Open destination dataset, set details
        dst_ds = driver.Create(dst_filename, nx, ny, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(proj)
        dst_ds.GetRasterBand(1).SetNoDataValue(nodatavalue)
        dst_ds.GetRasterBand(1).WriteArray(arr)

        # Close files
        dst_ds = None

    src_ds = None
    print "Finished year " + str(year)

print "FINISH"
