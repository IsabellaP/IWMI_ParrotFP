# read genIO CCI data and write it to csv timeseries
import numpy as np
from rsdata.ESA_CCI_SM.interface import ESA_CCI_SM
from pygeogrids.grids import CellGrid
import pygrids.ESA_CCI_SM as cci_grid
from datetime import datetime
from pytesmo.timedate import julian
import matplotlib.pyplot as plt


def read_gpis():
    
    fpath = '/media/sf_H/CCI_markus/coordinates.csv'
    coords = np.genfromtxt(fpath, dtype=[('name', 'S10'), ('lat', np.float), 
                                         ('lon', np.float)], delimiter=',',
                           skip_header=1)
    
    return coords


def CCI_genIO(valid_gpis, start_date, end_date, plot=False):
    
    start_jd = julian.julday(start_date.month, start_date.day, start_date.year, 
                             start_date.hour, start_date.minute, 
                             start_date.second)
    end_jd = julian.julday(end_date.month, end_date.day, end_date.year, 
                             end_date.hour, end_date.minute, 
                             end_date.second)
    
    parent_grid = cci_grid.ESA_CCI_SM_grid_v4_1_indl()
    nearest_gpis = parent_grid.find_nearest_gpi(valid_gpis['lon'], 
                                                valid_gpis['lat'])
    nearest_gpis = np.unique(nearest_gpis[0])
    cells = parent_grid.gpi2cell(nearest_gpis)
    
    header = 'jd,sm,sm_noise,sensor,freqband,nobs,year,month,day'
    descr = [('year', np.uint), ('month', np.uint), ('day', np.uint)]
    
    for cell in sorted(np.unique(cells)):
        gpis, lons, lats = parent_grid.grid_points_for_cell(cell)
        grid = CellGrid(lons, lats,
                        np.ones_like(lons, dtype=np.int16) * cell, gpis=gpis)
        
        cfg_path = ('/home/ipfeil/GitRepos/rs-data-readers/rsdata/'+
                    'ESA_CCI_SM/datasets/')
        version = 'ESA_CCI_SM_v02.3'
        param = 'esa_cci_sm_monthly'
        cci_io = ESA_CCI_SM(version=version, parameter=param, grid=grid,
                         cfg_path=cfg_path)
        
        for ts, gp in cci_io.iter_ts():
            if gp not in nearest_gpis:
                continue
            valid_date_idx = np.where((ts['jd']>=start_jd) & 
                                      (ts['jd']<=end_jd))[0]
            ts_valid_dates = ts[valid_date_idx]
            ts_dates = add_field(ts_valid_dates, descr)
            dates = julian.julian2datetime(ts_dates['jd'])
            years = [date.year for date in dates]
            ts_dates['year'] = years
            ts_dates['month'] = [date.month for date in dates]
            ts_dates['day'] = [date.day for date in dates]
            np.savetxt('/media/sf_D/CCI_csv/'+str(gp)+'.csv', 
                       ts_dates, delimiter=',', header=header)
            if plot == True:
                valid_ind = np.where(ts_valid_dates['sm'] != -999999)
                dates = julian.julian2datetime(ts_valid_dates['jd'][valid_ind])
                plt.plot(dates, ts_valid_dates['sm'][valid_ind])
                plt.title('ESA CCI SM combined monthly average, gpi: '+str(gp))
                plt.xlabel('date')
                plt.ylabel('soil moisture [%]')
                plt.show()
                

def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError, "'A' must be a structured numpy array"
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b
            

if __name__ == '__main__':
    valid_gpis = read_gpis()
    start_date = datetime(1991, 1, 1)
    end_date = datetime.today()
    CCI_genIO(valid_gpis, start_date, end_date, plot=False)
    
    print 'asdf'