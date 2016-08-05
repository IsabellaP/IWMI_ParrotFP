import matplotlib.pyplot as plt
import ascat
import pytesmo.io.sat.ers as ers
import os
#import rsdata.root_path as root
from rsdata.ESA_CCI_SM.interface import ESA_CCI_SM
import pygrids.ESA_CCI_SM as cci_quarter_grid
import pygrids.warp5 as warp_dgg
from pytesmo.timedate import julian
import pandas as pd
from datetime import datetime
from pytesmo.io.ismn.interface import ISMN_Interface
 

def read_ASCAT_ssm(gpi, plot=False): 
    ascat_folder = os.path.join(root.r, 'Datapool_processed', 'WARP', 'WARP5.5',
                                'IRMA1_WARP5.5_P2', 'R1', '080_ssm', 'netcdf')
    ascat_grid_folder = os.path.join(root.r,'Datapool_processed','WARP','ancillary',
                                     'warp5_grid')
    ascat_SSM_reader = ascat.AscatH25_SSM(ascat_folder, ascat_grid_folder,
                                         include_in_df=['sm', 'sm_noise', 'ssf', 
                                                        'proc_flag'])

    ascat_ts = ascat_SSM_reader.read_ssm(gpi)
    
    if plot == True:
        ascat_ts.plot()
        plt.show()
    
    return ascat_ts


def read_ERS_ssm(gpi, args=['sm', 'sm_noise', 'proc_flag', 'orbit_dir'], 
                 plot=False, start=None, end=None):
    
    ers_folder = os.path.join(root.r, 'Datapool_processed', 'WARP', 'WARP5.5',
                              'ERS_AMI_WS_WARP5.5_R1.1', '070_ssm', 'netcdf')
    
    ers_grid_folder = os.path.join(root.r,'Datapool_processed','WARP',
                                   'ancillary', 'warp5_grid')
    
    ers_SSM_reader = ers.ERS_SSM(ers_folder, ers_grid_folder,
                                 include_in_df=args)
    
    ers_df = ers_SSM_reader.read_ssm(gpi)
    ers_df = ers_df.data['sm']
    
    if start is not None and end is not None:
        ers_ts = ers_df[start:end]
    elif start is not None:
        ers_ts = ers_df[start:]
    elif end is not None:
        ers_ts = ers_df[:end]
    else:
        ers_ts = ers_df
        
    if plot == True:
        ers_ts.plot()
        plt.show()
    
    return ers_ts


def read_CCI(gpi, start=None, end=None):
    
    warp_grid = warp_dgg.DGGv21CPv20_ind_ld()
    cci_grid = cci_quarter_grid.ESA_CCI_SM_grid_v4_1_indl()
    
    cci_gpi = cci_grid.find_nearest_gpi(warp_grid.gpi2lonlat(gpi)[0], 
                                        warp_grid.gpi2lonlat(gpi)[1])[0]

    version = 'ESA_CCI_SM_v02.2'
    parameter = 'esa_cci_sm'
    cci_io = ESA_CCI_SM(version, parameter)
    
    ts = cci_io.read_ts(cci_gpi)            
    
    date_ind = julian.julian2datetime(ts['jd'])
    cci_df = pd.DataFrame(ts['sm'], index=date_ind, columns=['sm'])
    
    if start is not None and end is not None:
        cci_ts = cci_df[start:end]
    elif start is not None:
        cci_ts = cci_df[start:]
    elif end is not None:
        cci_ts = cci_df[:end]
    else:
        cci_ts = cci_df
        
    return cci_ts


def read_ISMN(plot=False):
    
    ismn_data_folder = os.path.join('.', 'data', 'Data_seperate_files_2011'+
                                    '1122_20121122_2364256_oqsd')
    ISMN_reader = ISMN_Interface(ismn_data_folder)

    network = 'IIT-KANPUR'
    station = 'IITK-Airstrip'
    station_obj = ISMN_reader.get_station(station)
    print "Available Variables at Station %s"%station
    #get the variables that this station measures
    variables = station_obj.get_variables()
    print variables
    
    depths_from,depths_to = station_obj.get_depths(variables[0])

    sensors = station_obj.get_sensors(variables[0],depths_from[0],depths_to[0])
    
    #read the data of the variable, depth, sensor combination
    time_series = station_obj.read_variable(variables[0],depth_from=depths_from[0],depth_to=depths_to[0],sensor=sensors[0])
    
    #print information about the selected time series
    print "Selected time series is:"
    print time_series
    #plot the data
    time_series.plot()
    #with pandas 0.12 time_series.plot() also works
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    # can be 'ascat', 'ers', 'cci', 'ismn'
    sm_dataset = 'cci'
    start = datetime(1991, 1, 1)
    end = datetime(2015, 12, 31)
    
    if sm_dataset == 'ismn':
        ismn_ts = read_ISMN()
    
    if sm_dataset == 'cci':
        gpi = 828158
        cci_ts = read_CCI(gpi, start, end)
    
    if sm_dataset == 'ers':
        gpis = [978129]#,
                #------------------------------------------------------ 1465021,
                #------------------------------------------------------ 1278825,
                #------------------------------------------------------ 1267293,
                #------------------------------------------------------ 1267293,
                #------------------------------------------------------ 1611813,
                #------------------------------------------------------ 1611813,
                #------------------------------------------------------- 516737,
                #------------------------------------------------------ 1117643,
                #------------------------------------------------------ 1384409,
                #------------------------------------------------------ 1314089,
                #------------------------------------------------------- 953549,
                #------------------------------------------------------ 1267293,
                #------------------------------------------------------ 1611813,
                #------------------------------------------------------ 1267293]
    
        for gpi in gpis:
            ers_ts = read_ERS_ssm(gpi, plot=True, start=start, end=end)
    
    print 'asdf'


