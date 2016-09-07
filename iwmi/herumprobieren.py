from simon import read_ts_area

path = 'C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\SWI\\SWI_stack.nc'
param = 'SWI'

lat_min = 19.204
lat_max = 21
lon_min = 74
lon_max = 76.5754
df = read_ts_area(path, param, lat_min, lat_max, lon_min, lon_max, t=10)

print "done"
