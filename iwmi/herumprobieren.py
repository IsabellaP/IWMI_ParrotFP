from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt

path = 'E:\\poets\\RAWDATA\\SWI_daily_stack\\SWI_daily_stack.nc'

with Dataset(path, 'r') as ncfile:
    data = ncfile.variables['SWI_040'][:,150,150]
    nctime = ncfile.variables['time'][:]
    unit_temps = ncfile.variables['time'].units
    try:
        cal_temps = ncfile.variables['time'].calendar
    except AttributeError:  # Attribute doesn't exist
        cal_temps = u"gregorian"  # or standard

    nc_all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)

plt.plot(nc_all_dates,data)
plt.show()

print 'done'
