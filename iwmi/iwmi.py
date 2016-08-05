import os
import iwmi_interface as ssm_iwmi
import read_ssm as ssm_TUW
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rsdata.root_path as root
from datetime import datetime
import pytesmo.temporal_matching as temp_match
import pytesmo.scaling as scaling
import pytesmo.metrics as metrics

def compare_ssm_stn(stn, warp_gpi, plot=False):
    
    df = ssm_iwmi.IWMI_read_csv()    
    iwmi_bare, iwmi_crop = ssm_iwmi.IWMI_ts_stn(df, stn)
    ers_ts = ssm_TUW.read_ERS_ssm(warp_gpi, args=['sm'])

    if plot == True:
        ax = ers_ts.data['sm'].plot(color='b')
        if len(iwmi_crop) != 0:
            iwmi_crop.plot(color='r', ax=ax)
        if len(iwmi_bare) != 0:
            iwmi_bare.plot(color='g', ax=ax)
        plt.legend(['ers ts', 'iwmi bare, index '+str(index), 
                    'iwmi crop, index '+str(index)])
        #plt.show()
        plt.savefig(os.path.join(root.x, 'staff', 'ipfeil', 'iwmi_plots',
                                 str(index)+'.png'))
        plt.clf()
    
    return iwmi_bare, iwmi_crop, ers_ts


def compare_ssm_index(index, warp_gpi, sm_dataset, start=None, end=None,
                      weekly=True, plot=False):
    
    df = ssm_iwmi.IWMI_read_csv()    
    iwmi_bare, iwmi_crop = ssm_iwmi.IWMI_ts_index(df, index)
    if sm_dataset == 'cci':
        sm_ts = ssm_TUW.read_CCI(warp_gpi, start, end)
    if sm_dataset == 'ascat':
        sm_ts = ssm_TUW.read_ASCAT_ssm(warp_gpi)
    if sm_dataset == 'ers':
        sm_ts = ssm_TUW.read_ERS_ssm(warp_gpi, args=['sm'], start=None,
                                     end=None)

    if weekly == True:        
        iwmi_bare_weekly = iwmi_bare.resample('W', how='mean').dropna()
        iwmi_crop_weekly = iwmi_crop.resample('W', how='mean').dropna()
        sm_ts_weekly = sm_ts.resample('W', how='mean').dropna()

        # correlation of bare and crop
        if len(iwmi_bare) != 0 and len(iwmi_crop) != 0:
            iwmi_bare_weekly.columns.values[0] = 1
            match_bare_crop = temp_match.matching(iwmi_crop_weekly, iwmi_bare_weekly)
            corr_crop_bare = metrics.spearmanr(match_bare_crop.iloc[:, 0], match_bare_crop.iloc[:, 1])[0]
            print corr_crop_bare

        if len(iwmi_bare) == 0:
            iwmi_bare_weekly = iwmi_bare_weekly
        else:
            match_bare = temp_match.matching(sm_ts_weekly, iwmi_bare_weekly)
            iwmi_bare_weekly_match = match_bare.iloc[:,1]
            sm_ts_weekly_bare = match_bare.iloc[:,0]
            iwmi_bare_resc = scaling.lin_cdf_match(iwmi_bare_weekly.iloc[:,0],
                                                   sm_ts_weekly)
            iwmi_bare_weekly = pd.DataFrame(iwmi_bare_resc,
                                            index=iwmi_bare_weekly.index)
            corr_bare = metrics.spearmanr(iwmi_bare_weekly_match, 
                                          sm_ts_weekly_bare)[0]

        if len(iwmi_crop) == 0:
            iwmi_crop_weekly = iwmi_crop_weekly
        else:
            match_crop = temp_match.matching(sm_ts_weekly, iwmi_crop_weekly)
            iwmi_crop_weekly_match = match_crop.iloc[:, 1]
            sm_ts_weekly_crop = match_crop.iloc[:,0]
            iwmi_crop_resc = scaling.lin_cdf_match(iwmi_crop_weekly.iloc[:,0],
                                                   sm_ts_weekly)
            iwmi_crop_weekly = pd.DataFrame(iwmi_crop_resc,
                                            index=iwmi_crop_weekly.index)
            corr_crop = metrics.spearmanr(iwmi_crop_weekly_match, 
                                          sm_ts_weekly_crop)[0]

    else:
        if len(iwmi_bare) == 0:
            iwmi_bare_resc = iwmi_bare
        else:
            iwmi_bare_resc = scaling.lin_cdf_match(iwmi_bare, sm_ts)
            
        if len(iwmi_crop) == 0:
            iwmi_crop_resc = iwmi_crop
        else:
            iwmi_crop_resc = scaling.lin_cdf_match(iwmi_crop, sm_ts)
    
        iwmi_bare = pd.DataFrame(iwmi_bare_resc,
                                index=iwmi_bare.index)
        iwmi_crop = pd.DataFrame(iwmi_crop_resc,
                                index=iwmi_crop.index)

    if plot == True:
        if weekly == True:
            sm_ts_plot = sm_ts_weekly
            iwmi_bare_plot = iwmi_bare_weekly
            iwmi_crop_plot = iwmi_crop_weekly
        else:
            sm_ts_plot = sm_ts
            iwmi_bare_plot = iwmi_bare
            iwmi_crop_plot = iwmi_crop
        ax = sm_ts_plot.plot(color='b')
        if len(iwmi_crop_plot) != 0:
            iwmi_crop_plot.plot(color='r', ax=ax)
        if len(iwmi_bare_plot) != 0:
            iwmi_bare_plot.plot(color='g', ax=ax)
        plt.legend([sm_dataset+' ts', 'iwmi crop, index '+str(index),
                    'iwmi bare, index '+str(index)])
        if sm_dataset in ['ers', 'ascat']:
            plt.ylabel('degree of saturation [%]')
            if 'corr_crop_bare' in locals():
                plt.title('corr_bare_crop ='+str(round(corr_crop_bare, 3)))
            plt.ylim([0, 140])
        else:
            plt.ylabel('volumetric soil moisture [m3/m3]')
            if 'corr_bare' in locals() and 'corr_crop' in locals():
                plt.title('corr_bare = '+str(round(corr_bare,3))+
                          '   corr_crop = '+str(round(corr_crop,3))+
                          '\n corr_bare_crop = '+str(round(corr_crop_bare,3)))
            elif 'corr_bare' in locals():
                plt.title('corr_bare = '+str(round(corr_bare,3)))
            elif 'corr_crop' in locals():
                plt.title('corr_crop = '+str(round(corr_crop,3)))
                plt.ylim([0, 100])
        plt.grid()
        #plt.show()
        plt.savefig(os.path.join(root.x, 'staff', 'ipfeil', 'iwmi_plots',
                                 sm_dataset,
                                 sm_dataset+'_cdf_'+str(index)+'.png'))
        plt.clf()
    
    return iwmi_bare_plot, iwmi_crop_plot, sm_ts_plot



if __name__ == "__main__":
    # read stations, indices and corresponding warp nearest neighbour
    path = os.path.join('.', 'auxiliary', 'iwmi_warp_nn.csv')
    dtype=np.dtype([('INDEX', np.int), ('STN', 'S4'),
                    ('DE', np.int), ('MI', np.int), 
                    ('DE.1', np.int), ('MI.1', np.int), ('lat', np.float), 
                    ('lon', np.float), ('WARP_nn', np.int), ('Maps', 'S15'), 
                    ('STN_full_name', 'S15')])
    data = np.genfromtxt(path, delimiter=',', dtype=dtype, skip_header=1)
    stations = data['STN']
    stn_index = np.unique(data['INDEX'])
    warp_gpis = data['WARP_nn']
    warp_gpis_index = np.unique(data['WARP_nn'])
    
    # mode can be 'index' or 'stn'
    ts_mode = 'index'
    # sm_dataset can be 'cci', 'ascat' or 'ers'. If cci, start and end can be
    # provided
    sm_dataset = 'ers'
    start = datetime(2001, 1, 1)
    end = datetime(2006, 12, 31)
    
    if ts_mode == 'stn':
        for idx in range(0, len(stations)):
            stn = stations[idx]
            warp_gpi = warp_gpis[idx]
            _, _, _ = compare_ssm_stn(stn, warp_gpi, plot=True)
    
    elif ts_mode == 'index':
        for idx in range(0, len(stn_index)):
            index = stn_index[idx]
            print(str(index))
            warp_gpi = warp_gpis_index[idx]
            _, _, _ = compare_ssm_index(index, warp_gpi, sm_dataset, start,
                                        end, plot=True)
    
    else:
        print('ts_mode must be either \'stn\' or \'index\'')
    
    print 'asdf'
