import os
import pandas as pd
from datetime import datetime
from veg_pred_readers import read_ts
import matplotlib.pyplot as plt
import pytesmo.temporal_matching as temp_match
import pytesmo.scaling as scaling
import pytesmo.df_metrics as df_metrics
import pytesmo.metrics as metrics


parrot_path = os.path.join('/media', 'sf_D', 'IWMI', 'Parrot', 'Kundasale.csv')
parrot_data = pd.read_csv(parrot_path)
parrot_index = pd.to_datetime(parrot_data['Capture Ts'], errors='coerce')
parrot_sm = pd.DataFrame(parrot_data['Vwc Percent'].values, 
                         columns=['parrot_sm'],
                         index=parrot_index.values)

swi_path = os.path.join('/media', 'sf_D', 'IWMI', '_DATA', 'SWI',
                        'SWI_daily_stack.nc')
# Kundasale
df_swi1, _, _ = read_ts(swi_path, params=['SWI_001'], lon=80.6948, lat=7.2903,
                  start_date=datetime(2016, 9, 23), 
                  end_date=datetime.today())

ax = parrot_sm.plot()
df_swi1.plot(ax=ax)
plt.title('Soil Moisture at field site Kundasale (Sri Lanka)')
plt.ylabel('[%]')
plt.grid()
plt.show()

matched_data = temp_match.matching(df_swi1,parrot_sm, window=1/24.)
 
scaled_data = scaling.scale(matched_data, method='lin_cdf_match',
                                         reference_index=1)

#now the scaled ascat data and insitu_sm are in the same space
scaled_data.plot(figsize=(15,5), title='Soil Moisture at field site Kundasale (Sri Lanka)')
plt.ylabel('[%]')
plt.grid()
plt.show()

label_ascat = 'SWI_001'
label_insitu = 'parrot_sm'

#calculate correlation coefficients, RMSD, bias, Nash Sutcliffe
x, y = scaled_data[label_ascat].values, scaled_data[label_insitu].values

print 'Results:'
#df_metrics takes a DataFrame as input and automatically
#calculates the metric on all combinations of columns
#returns a named tuple for easy printing
print df_metrics.pearsonr(scaled_data)
print "Spearman's (rho,p_value)", metrics.spearmanr(x, y)
print "Kendalls's (tau,p_value)", metrics.kendalltau(x, y)
print df_metrics.kendalltau(scaled_data)
print df_metrics.rmsd(scaled_data)
print "Bias", metrics.bias(x, y)
print "Nash Sutcliffe", metrics.nash_sutcliffe(x, y)

print 'done'
