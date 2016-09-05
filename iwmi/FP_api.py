# code from http://developer.parrot.com/docs/FlowerPower/?python#authentication
import os
from ConfigParser import SafeConfigParser
import requests
from pprint import pformat
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pytesmo.temporal_matching import matching
from pytesmo.scaling import scale
import pytesmo.metrics as metrics
from data_analysis import rescale_peng


def read_cfg(cfg_file, include_default=True, only_default=False):
    """
    Reads a WARP configuration file.

    Parameters
    ----------
    cfg_file : str
        File name of the configuration file.
    include_default : bool, optional
        Include default items. Default: True
    only_default : bool, optional
        Return only default items. Default: False

    Returns
    -------
    ds : dict
        Dictionary generated from the items and values in the config file.
    """
    if not os.path.isfile(cfg_file):
        raise IOError("Configuration file does not exist {:}".format(cfg_file))

    config = SafeConfigParser()
    config.read(cfg_file)

    cfg = {}

    if only_default:
        for item, value in config.defaults().iteritems():
            cfg[item] = eval_param(item, value)
    else:
        for section in config.sections():
            cfg[section] = {}
            for item, value in config.items(section):
                if include_default or item not in config.defaults().keys():
                    cfg[section][item] = value

    return cfg


def get_FP_data(username, password, client_id, client_secret, plant,
                print_results=True):
    """ Read data from flower power sensors.
    
    Parameters:
    ----------
    username, password, client_id, client_secret : str
        Credentials
    plant : str
        plant nickname
    print_results : bool
        if True, results are printed as long dictionary
        
    Returns:
    -------
    Flower power data
    """

    req = requests.get('https://apiflowerpower.parrot.com/user/v1/authenticate',
                       data={'grant_type': 'password',
                             'username': username,
                             'password': password,
                             'client_id': client_id,
                             'client_secret': client_secret,
                            })
    response = req.json()
    
    # Get authorization token from response
    access_token = response['access_token']
    auth_header = {'Authorization': 'Bearer {token}'.format(token=access_token)}
    
    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/user/v4/profile',
                       headers=auth_header)

    user = req.json()
    
    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/user/v1/versions',
                       headers=auth_header)

    versions = req.json()
    
    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v3/sync',
                            headers=auth_header)
    
    plants = req.json()

    # find plant and startdate in dictionary
    location_identifier = None
    possible_plants = []
    for i in range(len(plants['locations'])):
        possible_plants.append(plants['locations'][i]['plant_nickname'])
        if plants['locations'][i]['plant_nickname'] == plant:
            startdate = plants['locations'][i]['plant_assigned_date'].encode()
            startdate = datetime.strptime(startdate, '%Y-%m-%dT%H:%M:%SZ')
	    location_identifier = plants['locations'][i]['location_identifier']
            break
    
    if location_identifier is None:
        print 'Unknown plant. Chose one of '+str(possible_plants)
        return

    date_now = datetime.now()
    dates = []

    # create date range from startdate to datetime.now - delta is 10 days
    for delta in range(int((date_now - startdate).days/10.+1)):
        date = startdate + timedelta(days=10*delta)
        dates.append(date)

    dates.append(date_now)

    data = []
    # merge all dates in 10days steps till datetime.now
    for d in range(len(dates)-1):
        print('Read data for '+plant+ ' from '+str(dates[d])+
              ' to '+str(dates[d+1]))
        req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v2/sample/location/'
                       + location_identifier,
                       headers=auth_header,
                       params={'from_datetime_utc': dates[d],
                               'to_datetime_utc': dates[d+1]})
        samples = req.json()
        data += (samples['samples'])

    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v4/garden_locations_status',
                       headers=auth_header)
    
    garden = req.json()
    
    if print_results == True:
        # Access token etc.
        print('Server response: \n {0}'.format(pformat(response)))
        # user profile, Setting
        print('User Settings: \n {0}'.format(pformat(user)))
        # versions
        print('Versions: \n {0}'.format(pformat(versions)))
        # time series for given location
        print('Samples: \n {0}'.format(pformat(data)))
        # Plant data, plant settings
        print('Data and settings of each plant: \n {0}'.format(pformat(plants)))
        # time series for given location
        print('Samples: \n {0}'.format(pformat(samples)))
        # garden status for each location
        print('Garden status for each location: \n {0}'.format(pformat(garden)))
    
    return response, user, versions, data, plants, garden


def FPdata2df(samples, resample=None, path_out=None):
    """
    Convert data to nicer format (pd.DataFrame) and write to file, 
    perform resampling

    Parameters
    ----------
    samples : list of dictionaries
        flower power data
    resample : string
        rule for resampling e.g. 'H' for hourly, 'D' for daily
    path_out : string or None
        path for writing csv file

    Returns
    -------
    df : pd.DataFrame
        (resampled) flower power data as pd.DataFrame
    """

    # ignore fertilizer for now
    FP_data = samples
    if len(FP_data) == 0:
        print 'Empty dataset.'
        return
    ts = []
    data = []
    for i in range(len(FP_data)):
        date = FP_data[i]['capture_ts'].encode()
        date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
        ts.append(date)
        sm = FP_data[i]['vwc_percent']
        temp = FP_data[i]['air_temperature_celsius']
        light = FP_data[i]['par_umole_m2s']
        data.append([sm, temp, light])

    ts = np.array(ts)
    data = np.array(data)
    data[np.where(data[:, 0] > 100), 0] = 100

    df = pd.DataFrame(data=data, index=ts,
                      columns=['Parrot_vwc', 'air_temperature_celsius',
                               'par_umole_m2s'])
    if resample is not None:
        df = df.resample(resample).mean()

    if path_out is not None:
        df.to_csv(path_out)

    return df


def read_HOAL(path):
    """ Read HOAL soil moisture and temperature from upper layer (5 cm)"""
    
    hoal = pd.DataFrame()
    
    for hoal_file in sorted(os.listdir(path)):
        if hoal_file[0] == '.':
            continue
        hoal_data = pd.read_csv(os.path.join(path, hoal_file),
                                header=None, sep=';', skiprows=13,
                                names=['date', 'time', 'HOAL_sm0.05', 
                                       'HOAL_ts0.05'],
                                usecols=[0,1,2,7])
        hoal = hoal.append(hoal_data)

    date_idx=hoal['date']+' '+hoal['time']
    date_idx = [datetime.strptime(date, '%d.%m.%Y %H:%M:%S') 
                for date in date_idx]
    # apply calibration from Mariette
    sm_calib = hoal['HOAL_sm0.05'].values*0.559901289301 + 16.2029118301
    hoal_df = pd.DataFrame(sm_calib, index=date_idx, columns=['HOAL_sm0.05'])
    hoal_df['HOAL_ts0.05'] = hoal['HOAL_ts0.05'].values 
    
    return hoal_df


def read_HOAL_raw(path):
    """Read HOAL raw data"""
    
    columns = ['date', 'time', 'HOAL_raw_sm1']

    box22 = pd.read_csv(os.path.join(path, 'Box_022.dat'), delimiter='\t',
                        names=columns, decimal=',', usecols=[1,2,5],
                        parse_dates=[[0, 1]], dayfirst=True, skiprows=0)
    
    box22_df = pd.DataFrame(box22['HOAL_raw_sm1'].values, index=box22['date_time'], 
                            columns=['HOAL_raw_sm1'])

    return box22_df


def rescale_df(ascat_ssm, FP_df, hoal_df):
    ascat_ssm['ssm_ascat'] = ascat_ssm['ssm_ascat']*0.54
    #ascat_ssm.plot()
    #plt.show()
    
    matched_data = matching(ascat_ssm, FP_df['Parrot_vwc'], hoal_df)
    matched_data.plot()
    plt.title('Matched data: ASCAT, FP, HOAL')
    plt.show()
    
    scaled_data = scale(matched_data)#, method="mean_std")
    
    scaled_data.plot()
    plt.title('Satellite and in-situ soil moisture, HOAL Petzenkirchen')
    plt.ylabel('Volumetric Water Content [%]')
    plt.ylim([0,60])
    plt.show()


def calc_rho(ascat_ssm, FP_df, hoal_df):
    # multiply ASCAT with porosity (0.54) to get same units
    ascat_ssm['ssm_ascat'] = ascat_ssm['ssm_ascat']*0.54
    
    matched_data = matching(ascat_ssm, FP_df['Parrot_vwc'], 
                            hoal_df['HOAL_sm0.05'])
    matched_data.plot()
    plt.title('Matched data: ASCAT, FP, HOAL')
    plt.show()
    
    data_together = scale(matched_data)#, method="mean_std")
    
    ascat_rho = metrics.spearmanr(data_together['Parrot_vwc'].iloc[:-3], 
                                  data_together['ssm_ascat'].iloc[:-3])
    
    hoal_rho_sm = metrics.spearmanr(data_together['Parrot_vwc'].iloc[:-3], 
                                    data_together['HOAL_sm0.05'].iloc[:-3])
    
    exclude = ['HOAL_ts0.05', 'air_temperature_celsius', 'par_umole_m2s',
               'merge_key']
    data_together.ix[:, data_together.columns.difference(exclude)].plot()
    plt.title('Satellite and in-situ soil moisture, HOAL Petzenkirchen, station 22',
              fontsize=24)
              #+'\n rho_ASCAT_Parrot: '+str(np.round(ascat_rho[0],3))+
              #', rho_HOAL_Parrot: '+str(np.round(hoal_rho_sm[0],3)))
    plt.ylabel('Volumetric Water Content [%]',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.ylim([0,60])
    plt.show()


def plot_df(df, title_plant):
    
    df.plot()
    plt.ylim([0, 250])
    plt.title('ParrotFP '+title_plant)
    plt.xticks(rotation=30)
    plt.show()
    

if __name__ == '__main__':
    # read credentials, ascat and HOAL ssm - linux paths
    #cfg_path = '/media/sf_D/0_IWMI_DATASETS/FP_credentials.txt'
    #ascat_ssm = pd.DataFrame.from_csv('/media/sf_D/0_IWMI_DATASETS/ascat_ssm.csv')
    #hoal_raw = read_HOAL_raw('/media/sf_D/0_IWMI_DATASETS/HOAL_raw/')
    #hoal_df = read_HOAL('/media/sf_D/0_IWMI_DATASETS/HOAL/')
    
    # IWMI paths
    cfg_path = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\FP_credentials.txt'
    ascat_ssm = pd.DataFrame.from_csv('C:\\Users\\i.pfeil\\Documents\\'+
                                      '0_IWMI_DATASETS\\ascat_ssm.csv')
    #hoal_raw = read_HOAL_raw('C:\\Users\\i.pfeil\\Documents\\'+
    #                         '0_IWMI_DATASETS\\HOAL_raw\\')
    hoal_df = read_HOAL('C:\\Users\\i.pfeil\\Documents\\'+
                        '0_IWMI_DATASETS\\HOAL\\')
    
    # windows paths
    #cfg_path = 'D:\IWMI\FP_credentials.txt'
    #ascat_ssm = pd.DataFrame.from_csv('D:\IWMI\\ascat_SSM\\ascat_ssm.csv')
    
    cfg = read_cfg(cfg_path)
    cred = cfg['credentials']
    
    # possible plants: Balkonpflanze, Blumen, Lehmboden, Erdbeeren2, 
    # Sandige Erde, Petzenkirchen_Ackerrand, Dieffenbacchia, Erdbeeren
    plants = ['Petzenkirchen_Ackerrand']

    print_results = False

    for plant in plants:
        try:
            response, user, versions, samples, \
                        plants, garden = get_FP_data(cred['username'], 
                                                     cred['password'],
                                                     cred['client_id'],
                                                     cred['client_secret'],
                                                     plant, print_results)
            FP_df = FPdata2df(samples, resample='H', path_out=None)
            #plot_df(df, plant)
        except TypeError:
            break
    
    #===========================================================================
    # # bias raus??
    # end = np.where(FP_df['Parrot_vwc'] == 0)[0][0]
    # ascat_ub = ascat_ssm - ascat_ssm.mean()
    # FP_ub = FP_df['Parrot_vwc'].iloc[:end] - FP_df['Parrot_vwc'].iloc[:end].mean()
    # hoal_ub = hoal_df['HOAL_sm0.05'] - hoal_df['HOAL_sm0.05'].mean()
    # 
    # ascat = rescale_peng(ascat_ub, np.nanmin(ascat_ub), np.nanmax(ascat_ub))
    # FP = rescale_peng(FP_ub, np.nanmin(FP_ub), np.nanmax(FP_ub))
    # hoal = rescale_peng(hoal_ub, np.nanmin(hoal_ub), np.nanmax(hoal_ub))
    # matched_data = matching(ascat, FP, hoal)
    # 
    # matched_data.plot()
    # plt.show()
    #===========================================================================
    
    calc_rho(ascat_ssm, FP_df, hoal_df)
    
    print 'Finished'
