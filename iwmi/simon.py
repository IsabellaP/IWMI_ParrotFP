import os
import numpy as np
import pandas as pd
from datetime import datetime
from readers import read_img, read_ts, find_nearest
from netCDF4 import Dataset, num2date
from mpl_toolkits.basemap import Basemap
import fnmatch
import gdal
from pygeogrids.grids import BasicGrid
import matplotlib.pyplot as plt
import pytesmo.temporal_matching as temp_match
import pytesmo.metrics as metrics


def init_0_1_grid(str):
    '''
    Parameters:
    -----------
    str : str
        NDVI, SWI - which grid is needed
    Returns:
    --------
    gird : BasicGrid
    '''
    if str == 'SWI':
        fpath = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWI\\20070701"
        fname = "g2_BIOPAR_SWI10_200707010000_GLOBE_ASCAT_V3_0_1.nc"
        with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
            lon = ncfile.variables['lon'][:]
            lat = ncfile.variables['lat'][:]
            mask = (ncfile.variables["SWI_010"][:]).mask
        lons, lats = np.meshgrid(lon, lat)
        grid = BasicGrid(lons[np.where(mask == False)], lats[np.where(mask == False)])

    elif (str == 'NDVI') | (str == 'LAI'):
        fpath = "C:\\Users\\s.hochstoger\\Desktop\\poets\\DATA"
        fname = "West_SA_0.1_dekad_NDVI.nc"
        with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
            lon = ncfile.variables['lon'][:]
            lat = ncfile.variables['lat'][:]
            if str == 'NDVI':
                mask = (ncfile.variables['NDVI_dataset'][10]).mask
            else:
                mask = (ncfile.variables['NDVI_dataset'][29]).mask
        lons, lats = np.meshgrid(lon, lat)
        grid = BasicGrid(lons[np.where(mask == False)], lats[np.where(mask == False)])

    return grid

def read_ts_area(path, param, lat_min, lat_max, lon_min, lon_max, t=1):
    '''
    Reads all pixel of given area and returns the mean value per day
    for this area.
    Parameters:
    -----------
    path : str
        Path to stacked netcdf file
    param : str
        Parameter to be read (name as in nc-files).
    lat/lon_min/max : float, optional
        Bounding box coordinates, area to be read
    t : int, optional
        T-value of SWI, default=1
    Returns:
    --------
    data : pd.DataFrame
        Dataset containing mean value of whole area for each date
    '''
    with Dataset(path, "r") as ncfile:
        unit_temps = ncfile.variables['time'].units
        nctime = ncfile.variables['time'][:]
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError:  # Attribute doesn't exist
            cal_temps = u"gregorian"  # or standard

        all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)

        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]
        if param == 'SWI':
            param = 'SWI_' + str(t).zfill(3)
        mean = []
        dates = []
        for day in all_dates:
            nearest_date = find_nearest(all_dates, day)
            date_idx = np.where(all_dates == nearest_date)[0]

            lons = ncfile.variables['lon'][:]
            lats = ncfile.variables['lat'][:]
            lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
            lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
            data = ncfile.variables[param][date_idx, lat_idx, lon_idx]
            data[data < 0] = 0

            if np.ma.is_masked(data):
                mean_value = data.data[np.where(data.data != 255)].mean()
            else:
                mean_value = data.mean()
            mean.append(mean_value)
            dates.append(day)
            # if day.day == 24:
            #     dates.append(day.replace(day=23))
            # elif day.day == 22:
            #     dates.append(day.replace(day=23))
            # elif day.day == 21:
            #     dates.append(day.replace(day=23))
            # else:
            #     dates.append(day)

        data_df = {param: mean}
        df = pd.DataFrame(data=data_df, index=dates)
        if df.columns == 'SWI':
            df.columns = [param]

    return df


def anomaly(df):
    '''
    Calculates anomalies for time series. Of each day mean value of
    this day over all years is subtracted.
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    Returns:
    --------
    data : pd.DataFrame
        Dataset containing anomalies of input DataFrame
    '''
    group = df.groupby([df.index.month, df.index.day])
    m = {}
    df_anomaly = df.copy()
    for key, _ in group:
        m[key] = group.get_group(key).mean()

    for i in range(0, len(df_anomaly)):
        val = m[(df_anomaly.index[i].month, df_anomaly.index[i].day)]
        df_anomaly.iloc[i] = df_anomaly.iloc[i] - val

    col_str = df.columns[0] + ' Anomaly'
    df_anomaly.columns = [col_str]

    return df_anomaly


def plot_anomaly(df, df_anom):
    if df_anom.columns[0][0:3] == 'SWI':
        df_anom = df_anom / 100
        df = df / 100
        ax = df_anom.plot.area(stacked=False, figsize=[20, 15], color='b')
        df.plot(ax=ax, color='b')
        plt.title(df.columns[0] + ' and ' + df_anom.columns[0])
    else:
        ax = df_anom.plot.area(stacked=False, figsize=[20, 15], color='g')
        df.plot(ax=ax, color='g')
        plt.title(df.columns[0] + ' and ' + df_anom.columns[0])
    plt.grid()
    plt.axhline(0, color='black')
    plt.ylim([-0.5, 1])


def plot_area(lon_min, lon_max, lat_min, lat_max):
    path_to_nc_cop = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\VIs\\NDVI\\20070703'
    fname_cop = 'g2_BIOPAR_NDVI_200707030000_ASIA_VGT_V1_3.nc'
    with Dataset(os.path.join(path_to_nc_cop, fname_cop), mode='r') as ncfile:
        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]

    grid = np.meshgrid(lons, lats)
    cop_grid_lons = grid[0].flatten()
    cop_grid_lats = grid[1].flatten()

    index = np.where((cop_grid_lats <= lat_max) &
                     (cop_grid_lats >= lat_min) &
                     (cop_grid_lons <= lon_max) &
                     (cop_grid_lons >= lon_min))

    map = Basemap(projection='cyl', llcrnrlon=lons.min(), llcrnrlat=lats.min(), urcrnrlat=lats.max(),
                  urcrnrlon=lons.max())
    map.drawmapboundary()
    map.drawcoastlines()
    map.drawcountries()

    map.plot(cop_grid_lons[index], cop_grid_lats[index], marker='+', linewidth=0, color='m', markersize=5)


def plot_ts_anomalies(lat_min, lat_max, lon_min, lon_max):
    #======================== plot TS anomalies ===========
    ndvi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\NDVI_stack.nc"
    lai_path = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\LAI_stack.nc'
    fapar_path = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\FAPAR_stack.nc'
    swi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\SWI_stack.nc"

    plot_area(lon_min, lon_max, lat_min, lat_max)
    plt.savefig('C:\Users\s.hochstoger\Desktop\Plots\\' + 'area.png')

    df_ndvi = read_ts_area(ndvi_path, "NDVI", lat_min, lat_max, lon_min, lon_max)
    anomaly_ndvi = anomaly(df_ndvi)
    plot_anomaly(df_ndvi.loc[:'20140403'], anomaly_ndvi.loc[:'20140403'])
    plt.savefig('C:\Users\s.hochstoger\Desktop\Plots\\' + df_ndvi.columns[0] + '.png')

    df_lai = read_ts_area(lai_path, "LAI", lat_min, lat_max, lon_min, lon_max)
    anomaly_lai = anomaly(df_lai/7)
    plot_anomaly(df_lai/7, anomaly_lai)
    plt.savefig('C:\Users\s.hochstoger\Desktop\Plots\\' + df_lai.columns[0] + '.png')

    df_fapar = read_ts_area(fapar_path, "FAPAR", lat_min, lat_max, lon_min, lon_max)
    anomaly_fapar = anomaly(df_fapar)
    plot_anomaly(df_fapar.loc[:'20140403'], anomaly_fapar.loc[:'20140403'])
    plt.savefig('C:\Users\s.hochstoger\Desktop\Plots\\' + df_fapar.columns[0] + '.png')

    tt = [1, 5, 10, 15, 20, 40, 60, 100]
    for t in tt:
        df_swi = read_ts_area(swi_path, 'SWI_' + str(t).zfill(3), lat_min, lat_max, lon_min, lon_max)
        anomaly_swi = anomaly(df_swi)
        plot_anomaly(df_swi.loc[:'20140403'], anomaly_swi.loc[:'20140403'])
        plt.savefig('C:\Users\s.hochstoger\Desktop\Plots\\' + df_swi.columns[0] + '.png')


def calc_monthly_mean(param):
    grid = init_0_1_grid(param)
    gps = grid.get_bbox_grid_points(14.7148, 29.3655, 68.15, 81.8419)
    if param == 'NDVI':
        path = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\West_SA_0.1_dekad_NDVI.nc'
        means = _get_monthly_mean(path, param, grid, gps, t=None)
        means.to_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\monthly_mean_NDVI.csv")
    elif param == "LAI":
        path = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\West_SA_0.1_dekad_LAI.nc'
        means = _get_monthly_mean(path, param, grid, gps, t=None)
        means.to_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\monthly_mean_LAI.csv")
    elif param == 'SWI':
        t = [1, 5, 10, 15, 20, 40, 60, 100]
        path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\SWI_stack.nc"
        for tt in t:
            means = _get_monthly_mean(path, param, grid, gps, t=tt)
            means.to_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\monthly_mean_SWI_" + str(tt).zfill(3) + ".csv")
            print 'finished SWI ' + str(tt).zfill(3)
    return means

def _get_monthly_mean(path, param, grid, gpi, t=None):
    if param == 'SWI':
        param = param + '_' + str(t).zfill(3)
        startdate = datetime(2007, 7, 1)
        enddate = datetime(2016, 7, 1)
    elif param == 'LAI':
        param = param + '_dataset'
        startdate = datetime(2007, 7, 3)
        enddate = datetime(2013, 12, 3)
    elif param == 'NDVI':
        param = param + '_dataset'
        startdate = datetime(2007, 1, 1)
        enddate = datetime(2014, 5, 13)
    columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    df_mean = pd.DataFrame([], index=[-999], columns=columns)
    size = gpi.size
    for gp in gpi:
        lon, lat = grid.gpi2lonlat(gp)
        ts = read_ts(path, params=param, lon=lon, lat=lat, start_date=startdate,
                     end_date=enddate)
        ts[(ts == -99) | (ts == 255)] = np.NaN
        monthly = ts.resample('M').mean()
        group_month = monthly.groupby([monthly.index.month])
        mean = group_month.mean().transpose()

        mean = pd.DataFrame([mean.values[0]], index=[gp], columns=df_mean.columns)
        df_mean = df_mean.append(mean)

        done = (np.where(gpi == gp)[0][0])
        if done % 160 == 0:
            print done/float(size) * 100

    df_mean = df_mean[df_mean.index != -999]

    if param == 'SWI' + '_' + str(t).zfill(3):
        return df_mean
    elif param == 'NDVI_dataset':
        return df_mean/250
    elif param == 'LAI_dataset':
        return df_mean/210

def study_area_gpis():
    #=========STUDY AREA GRID POINTS ============
    gpi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\pointlist_India_warp.csv"
    gpis = pd.read_csv(gpi_path)
    gpi_path_p = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\pointlist_Pakistan_warp.csv"
    gpis_p = pd.read_csv(gpi_path_p)
    gpis = gpis.append(gpis_p)

    grid = BasicGrid(gpis.lon, gpis.lat, gpis.point)
    gpis = grid.get_bbox_grid_points(14.7148, 29.3655, 68.15, 81.8419)
    lon, lat = grid.get_bbox_grid_points(14.7148, 29.3655, 68.15, 81.8419, coords=True)
    gp = pd.DataFrame(gpis, columns=['gpi'])
    lon = pd.DataFrame(lon, columns=['lon'])
    lat = pd.DataFrame(lat, columns=['lat'])
    gplon = pd.concat([gp, lon], axis=1)
    gplonlat = pd.concat([gplon, lat], axis=1)
    gplonlat.to_csv("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\study_area_gp_lonlat_new.csv")


def read_img_new(path, param='NDVI', lat_min=5.9180, lat_max=9.8281,
             lon_min=79.6960, lon_max=81.8916, timestamp=datetime(2010, 7, 1),
             plot_img=False, swi='SWI_001'):
    """
    Parameters:
    -----------
    path : str
        Path to nc-data
    param : str, optional
        Parameter to be read (name as in nc-files). Default: NDVI
    lat/lon_min/max : float, optional
        Bounding box coordinates, default: Sri Lanka
    timestamp : datetime.datetime, optional
        Timestamp of image, default: 01/01/2014
    plot_img : bool, optional
        If true, result image is plotted, default: False
    Returns:
    --------
    data : dict
        Dataset
    """

    timestamp_array = []
    folders = os.listdir(path)

    if param == 'NDVI300':
        for fname in sorted(folders):
            year = int(fname[8:12])
            month = int(fname[12:14])
            day = int(fname[14:16])
            timestamp_array.append(datetime(year, month, day))

    else:  # NDVI, LAI, SWI
        for fname in sorted(folders):
            year = int(fname[0:4])
            month = int(fname[4:6])
            day = int(fname[6:8])
            timestamp_array.append(datetime(year, month, day))

    timestamp_array = np.array(timestamp_array)
    # find nearest timestamp
    nearest_date = find_nearest(timestamp_array, timestamp)
    date_idx = np.where(timestamp_array == nearest_date)[0]

    folder = np.array(sorted(folders))[date_idx][0]
    fpath = os.path.join(path, folder)
    fname = fnmatch.filter(os.listdir(fpath), '*.nc')[0]
    grid = init_0_1_grid('SWI')

    if param == 'SWI':
        # possible variables: SWI_001, 005, 010, 015, 020, 040, 060, 100
        key = swi
    elif param == 'NDVI300':
        key = 'NDVI'
    else:
        key = param

    with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]

        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        lon_data = lon[lon_idx]
        lat_data = lat[lat_idx]
        mesh = np.meshgrid(lon_data, lat_data)
        mesh = zip(mesh[0].flatten(), mesh[1].flatten())
        gps = []
        for point in mesh:
            gp = grid.find_nearest_gpi(point[0], point[1])[0]
            gps.append(gp)
        param_data = ncfile.variables[key][lat_idx, lon_idx]

    return param_data, gps, lon_data, lat_data


def read_poets_nc(poets_path, start_date, end_date, gpi=None, lon=None,
                  lat=None):

    if gpi is not None:
        grid = init_0_1_grid('NDVI')
        lon, lat = grid.gpi2lonlat(gpi)

    with Dataset(poets_path, "r") as ncfile:
        unit_temps = ncfile.variables['time'].units
        nctime = ncfile.variables['time'][:]
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError:  # Attribute doesn't exist
            cal_temps = u"gregorian"  # or standard

        timestamp = num2date(nctime, units=unit_temps, calendar=cal_temps)
        date_idx = np.where((timestamp >= start_date) &
                            (timestamp <= end_date))[0]

        # find nearest lonlat
        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]
        nearest_lon = find_nearest(lons, lon)
        nearest_lat = find_nearest(lats, lat)
        lon_idx = np.where(lons == nearest_lon)[0]
        lat_idx = np.where(lats == nearest_lat)[0]
        ndvi = ncfile.variables['NDVI_dataset'][date_idx, lat_idx, lon_idx]

    if np.ma.is_masked(ndvi):
        ndvi = ndvi.flatten().data
    else:
        ndvi = ndvi.flatten()

    ndvi[(ndvi == -99)] = np.NaN

    ndvi = pd.DataFrame(ndvi/250, columns=['NDVI'], index=timestamp[date_idx])

    return ndvi


def read_poets_nc_img(poets_path, date, lat_min, lat_max, lon_min, lon_max):

    grid = init_0_1_grid('LAI')
    with Dataset(poets_path, "r") as ncfile:
        unit_temps = ncfile.variables['time'].units
        nctime = ncfile.variables['time'][:]
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError:  # Attribute doesn't exist
            cal_temps = u"gregorian"  # or standard

        timestamp = num2date(nctime, units=unit_temps, calendar=cal_temps)
        nearest_date = find_nearest(timestamp, date)
        date_idx = np.where(timestamp == nearest_date)[0]

        # find nearest lonlat
        lons = ncfile.variables['lon'][:]
        lats = ncfile.variables['lat'][:]

        lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]

        lon_data = lons[lon_idx]
        lat_data = lats[lat_idx]
        mesh = np.meshgrid(lon_data, lat_data)
        mesh = zip(mesh[0].flatten(), mesh[1].flatten())
        gps = []
        for point in mesh:
            gp = grid.find_nearest_gpi(point[0], point[1])[0]
            gps.append(gp)
        ndvi = ncfile.variables['LAI_dataset'][date_idx, lat_idx, lon_idx]


    return ndvi, gps, lon_data, lat_data


def create_drought_dist(lat_min, lat_max, lon_min, lon_max):

    path = 'C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\IDSI_stack.nc'

    with Dataset(path, "r") as ncfile:
        unit_temps = ncfile.variables['time'].units
        nctime = ncfile.variables['time'][:]
        try:
            cal_temps = ncfile.variables['time'].calendar
        except AttributeError:  # Attribute doesn't exist
            cal_temps = u"gregorian"  # or standard

        all_dates = num2date(nctime, units=unit_temps, calendar=cal_temps)
    columns = ['healthy', 'watch', 'drought']
    df = pd.DataFrame([], index=[-999], columns=columns)
    for date in all_dates:
        data, _, _ = read_img(path, 'IDSI', lat_min, lat_max, lon_min, lon_max, timestamp=date)
        drought = np.where((data[0] == 1) | (data[0] == 2) | (data[0] == 3))[0].size
        watch = np.where((data[0] == 4) | (data[0] == 5))[0].size
        healthy = np.where((data[0] == 6) | (data[0] == 7))[0].size

        mean = pd.DataFrame([(healthy, watch, drought)], index=[date], columns=df.columns)
        df = df.append(mean)
    df = df[df.index != -999]
    df.index = all_dates
    df = df.divide(df.sum(axis=1), axis=0)
    #df.plot.area(alpha=0.5, ylim=(0, 1))

    int_steps = np.arange(43, 367, 46)
    for steps in int_steps:
        step_to = steps+6
        df.iloc[steps:step_to] = np.NAN
    df = df.interpolate('linear')

    return df

def plot_Droughts_and_Anomalies(lat_min, lat_max, lon_min, lon_max):

    df = create_drought_dist(lat_min, lat_max, lon_min, lon_max)
    df_swi = read_ts_area('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\SWI_stack.nc', 'SWI_040',
                          lat_min, lat_max, lon_min, lon_max)
    anomaly_swi = anomaly(df_swi)

    df_anom = anomaly_swi.loc[:'20150626'] / 100
    df_swi = df_swi.loc[:'20150626'] / 100
    df.drought = df.drought * (-1)

    ax = df.plot.area(alpha=0.4, figsize=[25, 10], color='gyr')
    df_anom.plot.area(ax=ax, stacked=False, color='b')
    df_swi.plot(ax=ax, color='b')
    plt.title('Drought Events in Maharashtra vs. Soil Water Index')
    plt.grid()
    plt.axhline(0, color='black')
    plt.ylim([-1, 1])
    #plt.savefig('C:\\Users\\s.hochstoger\\Desktop\\Plots\\IDSI_SWI040_Comparison_interpolate_newloc.png',
    #            dpi=450, bbox_inches='tight', pad_inches=0.3)


def calc_corr_IDSI_SWI(lat_min, lat_max, lon_min, lon_max):
    df = create_drought_dist(lat_min, lat_max, lon_min, lon_max)
    df.drought = df.drought * (-1)
    drought = pd.DataFrame(df.drought)
    tt = [1, 5, 10, 15, 20, 40, 60, 100]
    for t in tt:
        print t
        df_swi = read_ts_area('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\SWI_stack.nc',
                              'SWI_' + str(t).zfill(3), lat_min, lat_max, lon_min, lon_max)
        anomaly_swi = anomaly(df_swi)

        df_anom = anomaly_swi.loc[:'20150626'] / 100
        match = temp_match.matching(drought, df_anom)
        s_rho, s_p = metrics.spearmanr(match.iloc[:, 0], match.iloc[:, 1])
        print s_rho, s_p


if __name__ == '__main__':
    swi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\SWI_stack.nc"
    # #df_swi = read_ts_area(swi_path, 'SWI_040', 19.204, 21, 74, 76.5754)
    df_swi1 = read_ts(swi_path, params=['SWI_001'], lon=80.6948, lat=7.2903,
            start_date=datetime(2007, 7, 1), end_date=datetime(2016, 5, 31))
    df_swi2 = read_ts(swi_path, params=['SWI_001'], lon=80.4687, lat=8.1135,
            start_date=datetime(2007, 7, 1), end_date=datetime(2016, 5, 31))
    # anomaly_swi = anomaly(df_swi)
    # df_anom = anomaly_swi/100
    #
    # df_anom.plot.area(stacked=False, figsize=[20, 5], color='b', ylim=[-0.2, 0.2])
    # plt.axhline(-0.09, color='r', linewidth=2)
    # plt.axhline(-0.03, color='y', linewidth=2)
    # plt.axhline(0.03, color='limegreen', linewidth=2)
    # plt.axhline(0.09, color='darkgreen', linewidth=2)

    #calc_monthly_mean('SWI')
    #plot_Droughts_and_Anomalies(21.38, 22.30, 70, 72)
    #calc_corr_IDSI_SWI(21.38, 22.30, 70, 72)
    #plot_ts_anomalies(21.38, 22.30, 70, 72)
    #plot_ts_anomalies(21.204, 23, 75, 77.5754)
    #plot_Droughts_and_Anomalies(21.204, 23, 75, 77.5754)

    # ====== CCI ===========
    # ccits = read_ts('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\CCI_stack.nc', 'sm',
    #                75, 20, start_date=datetime(1978, 11, 1), end_date=datetime(2014, 12, 1))
    # cci = ccits[ccits != 255]
    # cci = cci.dropna(axis=0)
    # cci = read_ts_area('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\CCI_stack.nc', 'sm',
    #                   19.204, 21, 74, 76.5754)
    #
    #
    # years = np.arange(1978, 2015, 1)
    # months = np.arange(1, 13, 1)
    # days = [1, 10, 20]
    # dates = pd.DatetimeIndex([])
    # for year in years:
    #     for month in months:
    #         date = [datetime(year, month, 1), datetime(year, month, 10), datetime(year, month, 20)]
    #         index = pd.DatetimeIndex(date)
    #         dates = dates.append(index)
    # df_10 = pd.DataFrame([], index=dates, columns=['nan'])
    # cci_10 = cci.resample('10d').mean()
    # match = temp_match.matching(df_10, cci_10)
    # =========================================

    # ====== RAINFALL ===============
    # sm = read_ts_area('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\SWI_stack.nc', 'SWI_001',
    #                   21.38, 22.30, 70, 72)
    #
    # rf = read_ts_area('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\TRMM_RF_stack.nc', 'TRMM_RF',
    #                   21.38, 22.30, 70, 72)
    # rf_10d = rf.resample('10d').mean()
    #
    # match = temp_match.matching(sm, rf_10d)
    # s_rho, s_p = metrics.spearmanr(match.iloc[:, 0], match.iloc[:, 1])
    #
    # fig = plt.figure(figsize=[25, 10])
    # ax = fig.add_subplot(111)
    # lns1 = ax.plot(sm, label='Soil Moisture', color='g')
    # ax2 = ax.twinx()
    # lns2 = ax2.plot(rf.loc['20070701':], label='Rainfall', color='b')
    # ax.set_ylabel("Degree of Saturation [%]", fontsize=20)
    # ax2.set_ylabel("Rainfall [mm]", fontsize=20)
    # ax.set_ylim(0, 100)
    # ax2.set_ylim(0, 200)
    # lns = lns1 + lns2
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc=0, fontsize=20)
    # plt.savefig('C:\\Users\\s.hochstoger\\Desktop\\Plots\\SSM_Rainfall2.png',
    #             dpi=450, bbox_inches='tight', pad_inches=0.3)
    # rf_m = rf.resample('M').mean()
    #
    # anom1 = anomaly(rf)
    # anom2 = anomaly(rf_m)
    #
    # test = read_img('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\Dataset_stacks\\TRMM_RF_stack.nc', 'TRMM_RF',
    #                 14.7148, 29.3655, 68.15, 81.8419, timestamp=datetime(2013, 2, 2))
    # ====================================

    d=1
    # gg=init_0_1_grid('NDVI')
    # grid = init_0_1_grid('SWI')
    # star_date = datetime(2007, 1, 1)
    # end_date = datetime(2014, 5, 13)
    # ndvi = read_poets_nc("C:\Users\s.hochstoger\Desktop\poets\DATA\West_SA_0.1_dekad.nc",
    #                      star_date, end_date, lon=75, lat=20)

    # fpath = os.path.join('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWI\\', '20070711')
    # fname = 'g2_BIOPAR_SWI10_200707110000_GLOBE_ASCAT_V3_0_1.nc'
    # with Dataset(os.path.join(fpath, fname), mode='r') as ncfile:
    #     lons = ncfile.variables['lon'][:]
    #     lats = ncfile.variables['lat'][:]


    #=============== map SWI anomalies for study area
    # swi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\SWI\\"
    # years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
    # months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # days = [3, 13, 23]
    # all_means = pd.read_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\monthly_mean_SWI_040.csv')
    # all_means.index = all_means.iloc[:, 0].values
    # all_means = all_means.drop('Unnamed: 0', 1)
    #
    # for year in years:
    #     for month in months:
    #
    #         data1, gp1, lon1, lat1 = read_img_new(swi_path, 'SWI', 14.7148, 29.3655, 68.15, 81.8419,
    #                                           timestamp=datetime(year, month, days[0]), swi='SWI_040')
    #         data2, gp2, lon2, lat2 = read_img_new(swi_path, 'SWI', 14.7148, 29.3655, 68.15, 81.8419,
    #                                           timestamp=datetime(year, month, days[1]), swi='SWI_040')
    #         data3, gp3, lon3, lat3 = read_img_new(swi_path, 'SWI', 14.7148, 29.3655, 68.15, 81.8419,
    #                                           timestamp=datetime(year, month, days[2]), swi='SWI_040')
    #         array_all = np.ma.vstack(data1, data2, data3)
    #         mean = np.ma.mean(array_all, axis=0)
    #         df = pd.DataFrame(mean.data.flatten(), index=gp1, columns=[str(month)])
    #         df.iloc[np.where(df == 0)[0]] = np.NAN
    #
    #         anom = df-all_means
    #         anom = anom.dropna(axis=0, how='all')
    #         anom = anom.dropna(axis=1, how='all')
    #
    #         grid = init_0_1_grid('SWI')
    #         lon_anom = []
    #         lat_anom = []
    #         for gp in anom.index.values:
    #             lon, lat = grid.gpi2lonlat(gp)
    #             lon_anom.append(lon)
    #             lat_anom.append(lat)
    #
    #         plt.figure(figsize=(20, 15))
    #         map = Basemap(projection='cyl', llcrnrlon=68.14, llcrnrlat=14.71, urcrnrlat=29.37,
    #                       urcrnrlon=81.85)
    #         map.drawmapboundary()
    #         map.drawcoastlines()
    #         map.drawcountries()
    #         #map.readshapefile("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\IND_adm1\\IND_adm1", 'IND_adm1')
    #         cm = plt.cm.get_cmap('RdYlBu')
    #         map.scatter(np.array(lon_anom).flatten(), np.array(lat_anom).flatten(), c=np.array(anom).flatten(),
    #                     edgecolor='None', marker='s', s=75, vmin=-30, vmax=30, cmap=cm)
    #         plt.colorbar()
    #         plt.title('SWI Anomalies ' + str(year) + '_' + str(month).zfill(2) + ' T = 40', fontsize=21)
    #         plt.savefig('C:\\Users\\s.hochstoger\\Desktop\\Plots\\' + str(year) + '_' + str(month).zfill(2) + '_anomalies_T40.png',
    #                     dpi=450, bbox_inches='tight', pad_inches=0.3)
    #=========================================

    # #=============== map NDVI anomalies for study area
    # ndvi_path = "C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\West_SA_0.1_dekad_LAI.nc"
    # years = [2008, 2009, 2010, 2011, 2012, 2013]
    # months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # days = [10, 20, 27]
    # all_means = pd.read_csv('C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\monthly_mean_LAI.csv')
    # all_means.index = all_means.iloc[:, 0].values
    # all_means = all_means.drop('Unnamed: 0', 1)
    #
    # for year in years:
    #     for month in months:
    #
    #         data1, gp1, lon1, lat1 = read_poets_nc_img(ndvi_path, datetime(year, month, days[0]),
    #                                                    14.7148, 29.3655, 68.15, 81.8419)
    #         data2, gp2, lon2, lat2 = read_poets_nc_img(ndvi_path, datetime(year, month, days[1]),
    #                                                    14.7148, 29.3655, 68.15, 81.8419)
    #         data3, gp3, lon3, lat3 = read_poets_nc_img(ndvi_path, datetime(year, month, days[2]),
    #                                                    14.7148, 29.3655, 68.15, 81.8419)
    #         array_all = np.ma.vstack(data1, data2, data3)
    #         mean = np.ma.mean(array_all, axis=0)
    #         df = pd.DataFrame((mean.data/210).flatten(), index=gp1, columns=[str(month)])
    #         df.iloc[np.where(df == 0)[0]] = np.NAN
    #         df = df.dropna(axis=0, how='all')
    #
    #         anom = df-all_means
    #         anom = anom.dropna(axis=0, how='all')
    #         anom = anom.dropna(axis=1, how='all')
    #
    #         grid = init_0_1_grid('LAI')
    #         lon_anom = []
    #         lat_anom = []
    #         for gp in anom.index.values:
    #             lon, lat = grid.gpi2lonlat(gp)
    #             lon_anom.append(lon)
    #             lat_anom.append(lat)
    #
    #         plt.figure(figsize=(20, 15))
    #         map = Basemap(projection='cyl', llcrnrlon=68.14, llcrnrlat=14.71, urcrnrlat=29.37,
    #                       urcrnrlon=81.85)
    #         map.drawmapboundary()
    #         map.drawcoastlines()
    #         map.drawcountries()
    #         #map.readshapefile("C:\\Users\\s.hochstoger\\Desktop\\0_IWMI_DATASETS\\IND_adm1\\IND_adm1", 'IND_adm1')
    #         cm = plt.cm.get_cmap('RdYlGn')
    #         map.scatter(np.array(lon_anom).flatten(), np.array(lat_anom).flatten(), c=np.array(anom).flatten(),
    #                     edgecolor='None', marker='s', s=70, vmin=-0.30, vmax=0.30, cmap=cm)
    #         plt.colorbar()
    #         plt.title('LAI Anomalies ' + str(year) + '_' + str(month).zfill(2), fontsize=21)
    #         plt.savefig('C:\\Users\\s.hochstoger\\Desktop\\Plots\\LAI_Spatial_Anomalies_monthly\\'
    #                     + str(year) + '_' + str(month).zfill(2) + '_anomalies.png', dpi=450,
    #                     bbox_inches='tight', pad_inches=0.3)
    #=========================================

    pass
