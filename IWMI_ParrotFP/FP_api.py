# code from http://developer.parrot.com/docs/FlowerPower/?python#authentication

import requests
from pprint import pformat  # here only for aesthetic

# First we set our credentials
username = 'isabella.pfeil@gmail.com'
password = 'magnolia2'

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
print('Server response: \n {0}'.format(pformat(response)))

# Get authorization token from response
access_token = response['access_token']
auth_header = {'Authorization': 'Bearer {token}'.format(token=access_token)}

# From now on, we won't need initial credentials: access_token and auth_header will be enough.

# Set your own authentication token
req = requests.get('https://apiflowerpower.parrot.com/user/v4/profile',
                   headers={'Authorization': 
                            'Bearer {token}'.format(token=access_token)})


response = req.json()
print('Server response: \n {0}'.format(pformat(response)))

# Set your own authentication token
req = requests.get('https://apiflowerpower.parrot.com/user/v1/versions',
                   headers={'Authorization': 
                            'Bearer {token}'.format(token=access_token)})


response = req.json()
print('Server response: \n {0}'.format(pformat(response)))


location_identifier = 'eg9dnEtyHy589153XXX'

# Set your own authentication token
req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v2/sample/location/' 
                   + location_identifier,
                   headers={'Authorization': 
                            'Bearer {token}'.format(token=access_token)})
    #params={'from_datetime_utc': '2016-06-18T14:42:42Z',
     #       'to_datetime_utc': '2016-06-19T06:30:00Z'})

response = req.json()
print('Server response: \n {0}'.format(pformat(response)))

# Set your own authentication token
req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v3/sync',
                        headers={'Authorization':
                            'Bearer {token}'.format(token=access_token)})

response = req.json()
print('Server response: \n {0}'.format(pformat(response)))

# Set your own authentication token
req = requests.get('https://apiflowerpower.parrot.com/sensor_data/v4/garden_locations_status',
                   headers={'Authorization': 
                            'Bearer {token}'.format(token=access_token)})

response = req.json()
print('Server response: \n {0}'.format(pformat(response)))