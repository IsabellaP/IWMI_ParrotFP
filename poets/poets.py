import os
from datetime import datetime
from poets.poet import Poet

# poets attributes:
rootpath = os.path.join('D:\\', 'Test')
regions = ['AU'] # clipping to Austria
spatial_resolution = 0.1
temporal_resolution = 'dekad'
start_date = datetime(2000, 1, 1)
nan_value = -99

# initializing Poet class:
p = Poet(rootpath, regions, spatial_resolution, temporal_resolution,
         start_date, nan_value)