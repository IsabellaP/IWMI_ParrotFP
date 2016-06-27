# code from http://developer.parrot.com/docs/FlowerPower/?python#authentication

import requests
from pprint import pformat  # here only for aesthetic
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def get_FP_data(plant, start_date, end_date, print_results=True):
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

    if plant == 'erdbeeren':
        location_identifier = 'vTPyGmPsbl1466103276199'
    elif plant == 'dieffenbacchia':
        location_identifier = 'LBFxUGtLcT1465927113542'
    elif plant == 'petzenkirchen':
        location_identifier = 'Y3QjXXEQmN1466537309325'
    elif plant == 'erdbeeren2':
        location_identifier = 'Enww4lXuIL1466518968168'
    
    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v2/sample/location/' 
                       + location_identifier,
                       headers={'Authorization': 
                                'Bearer {token}'.format(token=access_token)},
                       params={'from_datetime_utc': start_date,
                               'to_datetime_utc': end_date})
    
    samples = req.json()
    
    # Set your own authentication token
    req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v3/sync',
                            headers={'Authorization':
                                'Bearer {token}'.format(token=access_token)})
    
    plants = req.json()
    
    
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
        print('Samples: \n {0}'.format(pformat(samples)))
        # Plant data, plant settings
        print('Data and settings of each plant: \n {0}'.format(pformat(plants)))
        # garden status for each location
        print('Garden status for each location: \n {0}'.format(pformat(garden)))
    
    return response, user, versions, samples, plants, garden


def FPdata2df(samples, path_out=None):
    
    # ignore fertilizer for now
    FP_data = samples['samples']
    if len(FP_data) == 0:
        print 'Empty dataset.'
        return
    ts = []
    data = []
    for i in range(len(FP_data)): 
        ts.append(FP_data[i]['capture_ts'])
        sm = FP_data[i]['vwc_percent']
        temp = FP_data[i]['air_temperature_celsius']
        light = FP_data[i]['par_umole_m2s']
        data.append([sm, temp, light])
        
    ts = np.array(ts)
    data = np.array(data)
    
    df = pd.DataFrame(data=data, index=ts, 
                      columns=['vwc_percent', 'air_temperature_celsius',
                               'par_umole_m2s'])
    
    print df
    if path_out is not None:
        df.to_csv(path_out)
    
    return df
    

def plot_df(df1, df2):
    
    df1.plot()
    plt.ylim([0,250])
    plt.title('ParrotFP1 - Strawberries')
    plt.xticks(rotation=30)
    df2.plot()
    plt.ylim([0,250])
    plt.title('ParrotFP2 - Strawberries')
    plt.xticks(rotation=30)
    plt.show()


if __name__ == '__main__':
    
    plants = ['erdbeeren', 'erdbeeren2']
    # add 2 to get UTC+2
    start_date = '2016-06-22T03:00:00Z'
    end_date = '2016-06-24T07:00:00Z'
    print_results = False
    
    response, user, versions, samples1, \
                    plants, garden = get_FP_data('erdbeeren', start_date,
                                                 end_date, print_results)
    response, user, versions, samples2, \
                    plants, garden = get_FP_data('erdbeeren2', start_date,
                                                 end_date, print_results)
    timestamp = datetime.now()
    
    #path_out = '/data/ParrotFP/ParrotFP_'+plant+str(timestamp)+'.csv'
    df1 = FPdata2df(samples1, path_out=None)
    df2 = FPdata2df(samples2, path_out=None)
    
    plot_df(df1, df2)
    
    print 'Finished'

