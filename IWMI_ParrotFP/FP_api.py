# code from http://developer.parrot.com/docs/FlowerPower/?python#authentication

import requests
from pprint import pformat  # here only for aesthetic
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#import pytesmo.temporal_matching as temp_match
#import pytesmo.scaling as scaling
#import pytesmo.metrics as metrics


def get_FP_data(plant, print_results=True):
    # First we set our credentials
    username = 'isabella.pfeil@gmail.com'
    password = 'flowerpower1'
    
    #from the developer portal
    client_id = 'isabella.pfeil@gmail.com'
    client_secret = 'SONKzgRvZ2uKv1lb8LI7qBDE7JBw3M81FX10jDQq6F4oHtGm'
    
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
    
    # From now on, we won't need initial credentials: access_token and auth_header will be enough.
    
    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/user/v4/profile',
                       headers={'Authorization': 
                                'Bearer {token}'.format(token=access_token)})
    
    
    user = req.json()
    
    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/user/v1/versions',
                       headers={'Authorization': 
                                'Bearer {token}'.format(token=access_token)})
    
    
    versions = req.json()    

    if plant == 'Erdbeeren':
        location_identifier = 'vTPyGmPsbl1466103276199'
    elif plant == 'Dieffenbacchia':
        location_identifier = 'LBFxUGtLcT1465927113542'
    elif plant == 'Petzenkirchen_Ackerrand':
        location_identifier = 'Y3QjXXEQmN1466537309325'
    elif plant == 'Erdbeeren2':
        location_identifier = 'Enww4lXuIL1466518968168'
    elif plant == 'Balkonpflanze':
        location_identifier = '4EUPOCN4yH1467128358630'
    elif plant == 'Sandige Erde':
        location_identifier = 'slvfkXyzFV1467128115063'


    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v3/sync',
                            headers={'Authorization':
                                'Bearer {token}'.format(token=access_token)})

    plants = req.json()

    # find plant and startdate in dictionary
    for i in range(len(plants['locations'])):
        if plants['locations'][i]['plant_nickname'] == plant:
            startdate = plants['locations'][i]['plant_assigned_date'].encode()
            startdate = datetime.strptime(startdate, '%Y-%m-%dT%H:%M:%SZ')
            break

    enddate = startdate + timedelta(days=10)
    timestamp = datetime.now()
    dates = []

    # create date range from startdate to datetime.now - delta is 10 days
    for delta in range(int((timestamp - startdate).days/10.+1)):
        date = startdate + timedelta(days=10*delta)
        dates.append(date)

    dates.append(timestamp)

    data = []
    # merge all dates in 10days steps till datetime.now
    for d in range(len(dates)-1):
        req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v2/sample/location/'
                       + location_identifier,
                       headers={'Authorization': 
                                'Bearer {token}'.format(token=access_token)},
                       params={'from_datetime_utc': dates[d],
                               'to_datetime_utc': dates[d+1]})
        samples = req.json()
        data += (samples['samples'])


    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v4/garden_locations_status',
                       headers={'Authorization': 
                                'Bearer {token}'.format(token=access_token)})
    
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
        # garden status for each location
        print('Garden status for each location: \n {0}'.format(pformat(garden)))
    
    return response, user, versions, data, plants, garden


def FPdata2df(samples, resample=None, path_out=None):

    '''
    .....

    Parameters
    ----------
    samples :

    resample : string
        rule for resampling e.g. 'H' for hourly, 'D' for daily
    path_out : string or None
        path for writing csv file


    Returns
    -------

    '''

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
        sm = FP_data[i]['vwc_percent']*2
        temp = FP_data[i]['air_temperature_celsius']
        light = FP_data[i]['par_umole_m2s']
        data.append([sm, temp, light])

    ts = np.array(ts)
    data = np.array(data)
    data[np.where(data[:, 0] > 100), 0] = 100

    df = pd.DataFrame(data=data, index=ts,
                      columns=['vwc_percent', 'air_temperature_celsius',
                               'par_umole_m2s'])
    if resample is not None:
        df.resample(resample, how='mean')

    print df

    if path_out is not None:
        df.to_csv(path_out)

    return df

def plot_df(df1, df2):

    df1.plot()
    plt.ylim([0, 250])
    plt.title('ParrotFP1 - Strawberries')
    plt.xticks(rotation=30)
    df2.plot()
    plt.ylim([0, 250])
    plt.title('ParrotFP2 - Strawberries')
    plt.xticks(rotation=30)
    plt.show()
    

def calc_corr(df1, df2):
    pass
    
    #------- df_sm = pd.DataFrame(data=[df1['vwc_percent'], df2['vwc_percent']],
                         #------------------------------------- index=df1.index,
                         #------------------------------ columns=['sm1', 'sm2'])
    #--------------------------------------- corr = df_sm.corr(method='pearson')
#------------------------------------------------------------------------------ 
    #--------------------------------------------------------------- return corr


if __name__ == '__main__':

    plants = ['Balkonpflanze', 'Sandige Erde']
    # add 2 to get UTC+2
    start_date = '2016-07-02T08:00:00Z'
    end_date = '2016-07-12T08:00:00Z'

    print_results = False
    
    response, user, versions, samples1, \
                    plants, garden = get_FP_data('Balkonpflanze', print_results)
    response, user, versions, samples2, \
                    plants, garden = get_FP_data('Petzenkirchen_Ackerrand', print_results)
    timestamp = datetime.now()
    
    #path_out = '/data/ParrotFP/ParrotFP_'+plant+str(timestamp)+'.csv'
    df1 = FPdata2df(samples1, resample='H', path_out=None)
    df2 = FPdata2df(samples2, path_out=None)
    corr = calc_corr(df1, df2)
    
    plot_df(df1, df2)
    
    print 'Finished'

