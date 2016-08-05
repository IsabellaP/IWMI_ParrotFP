import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def IWMI_write_csv(sm_path=None, path_out=None):
    
    if sm_path is None:
        sm_path = os.path.join('.', 'data', 'sm_edit.txt')
    # changes in sm_edit:
    # 42675 2002 06 18   0.0 10.5...: rh2: 001 statt 0 1 (gab Fehler)
    if path_out is None:
        path_out = os.path.join('.', 'data', 'sm_dataframe.csv')
    
    with open(sm_path) as f:
        index = []
        year = []
        mn = []
        dt = []
        rf = []
        evp = []
        et = []
        ssh = []
        rh1 = []
        rh2 = []
        maxi = []
        mini = []
        mtmp = []
        wsm1_bare = []
        wsm2_bare = []
        wsm3_bare = []
        wsm4_bare = []
        wsm5_bare = []
        wsm6_bare = []
        wsm1_crop = []
        wsm2_crop = []
        wsm3_crop = []
        wsm4_crop = []
        wsm5_crop = []
        wsm6_crop = []
        wk = []
        di = []
        stn = []
        lat_de = []
        lat_mi = []
        lon_de = []
        lon_mi = []
        t = []
        name_var_of_crop = []
        sowing = []
        harvest = []
        
        lines = f.readlines()[43:-2]
        idx = 0
        while idx < len(lines):
            print idx
            line = lines[idx]
            if line.startswith('-'):
                idx += 4
                print '-'
                continue
            if line.startswith('\x0cSOIL'):
                idx += 5
                print 'SOIL'
                continue
            index.append(line[0:5])
            year.append(line[6:10])
            mn.append(line[11:13])
            dt.append(line[14:16])
            rf.append(line[17:22])
            evp.append(line[23:27])
            et.append(line[28:32])
            ssh.append(line[33:37])
            rh1.append(line[38:41])
            rh2.append(line[42:45])
            maxi.append(line[46:51])
            mini.append(line[52:57])
            mtmp.append(line[58:63])
            wsm1_bare.append(line[64:68])
            wsm2_bare.append(line[69:73])
            wsm3_bare.append(line[74:78])
            wsm4_bare.append(line[79:83])
            wsm5_bare.append(line[84:88])
            wsm6_bare.append(line[89:93])
            wsm1_crop.append(line[94:98])
            wsm2_crop.append(line[99:103])
            wsm3_crop.append(line[104:108])
            wsm4_crop.append(line[109:113])
            wsm5_crop.append(line[114:118])
            wsm6_crop.append(line[119:123])
            wk.append(line[124:126])
            di.append(line[127:129])
            stn.append(line[130:134])
            lat_de.append(line[135:137])
            lat_mi.append(line[138:140])
            lon_de.append(line[141:143])
            lon_mi.append(line[144:146])
            t.append(line[147:148])
            name_var_of_crop.append(line[149:168])
            sowing.append(line[170:176])
            harvest.append(line[177:184])
            idx += 1
    
    variables = [index, year, mn, dt, rf, evp, et, ssh, rh1, rh2, maxi, mini, mtmp,
                 wsm1_bare, wsm2_bare, wsm3_bare, wsm4_bare, wsm5_bare, wsm6_bare,
                 wsm1_crop, wsm2_crop, wsm3_crop, wsm4_crop, wsm5_crop, wsm6_crop,
                 wk, di, stn, lat_de, lat_mi, lon_de, lon_mi, t, name_var_of_crop,
                 sowing, harvest]
    
    for var in variables:
        for idx, element in enumerate(var):
            element = element.strip()
            if element == '':
                var[idx] = -9999
    
    index = np.array(map(int, index))
    year = np.array(map(int, year))
    mn = np.array(map(int, mn))
    dt = np.array(map(int, dt))
    rf = np.array(map(float, rf))
    evp = np.array(map(float, evp))
    et = np.array(map(float, et))
    ssh = np.array(map(float, ssh))
    rh1 = np.array(map(int, rh1))
    rh2 = np.array(map(int, rh2))
    maxi = np.array(map(float, maxi))
    mini = np.array(map(float, mini))
    mtmp = np.array(map(float, mtmp))
    wsm1_bare = np.array(map(float, wsm1_bare))
    wsm2_bare = np.array(map(float, wsm2_bare))
    wsm3_bare = np.array(map(float, wsm3_bare))
    wsm4_bare = np.array(map(float, wsm4_bare))
    wsm5_bare = np.array(map(float, wsm5_bare))
    wsm6_bare = np.array(map(float, wsm6_bare))
    wsm1_crop = np.array(map(float, wsm1_crop))
    wsm2_crop = np.array(map(float, wsm2_crop))
    wsm3_crop = np.array(map(float, wsm3_crop))
    wsm4_crop = np.array(map(float, wsm4_crop))
    wsm5_crop = np.array(map(float, wsm5_crop))
    wsm6_crop = np.array(map(float, wsm6_crop))
    wk = np.array(map(int, wk))
    di = np.array(map(int, di))
    stn = np.array(stn)
    lat_de = np.array(map(int, lat_de))
    lat_mi = np.array(map(int, lat_mi))
    lon_de = np.array(map(int, lon_de))
    lon_mi = np.array(map(int, lon_mi))
    t = np.array(t)
    name_var_of_crop = np.array(name_var_of_crop)
    sowing = np.array(map(int, sowing))
    harvest = np.array(map(int, harvest))
    
    df = pd.DataFrame()
    df['INDEX']=index
    df['YEAR']=year
    df['MN']=mn
    df['DT']=dt
    df['RF']=rf
    df['EVP']=evp
    df['ET']=et
    df['SSH']=ssh
    df['RH1']=rh1
    df['RH2']=rh2
    df['MAX']=maxi
    df['MIN']=mini
    df['MTMP']=mtmp
    df['WSM1']=wsm1_bare
    df['WSM2']=wsm2_bare
    df['WSM3']=wsm3_bare
    df['WSM4']=wsm4_bare
    df['WSM5']=wsm5_bare
    df['WSM6']=wsm6_bare
    df['WSM1.1']=wsm1_crop
    df['WSM2.1']=wsm2_crop
    df['WSM3.1']=wsm3_crop
    df['WSM4.1']=wsm4_crop
    df['WSM5.1']=wsm5_crop
    df['WSM6.1']=wsm6_crop
    df['WK']=wk
    df['DI']=di
    df['STN']=stn
    df['DE']=lat_de
    df['MI']=lat_mi
    df['DE.1']=lon_de
    df['MI.1']=lon_mi
    df['T']=t
    df['NAME_VAR_OF_CROP']=name_var_of_crop
    df['SOWING']=sowing
    df['HARVEST']=harvest
    
    df.to_csv(path_out)
    

def IWMI_read_csv(path=None):
    
    if path is None:
        path = os.path.join('.', 'data', 'sm_dataframe.csv')
        
    data = pd.read_csv(path)
    
    return data


def IWMI_ts_stn(df, stn, plot=False):

    stn_ind = np.where((df['STN'] == stn+str(' ')) | (df['STN'] == str(' ')+stn) |
                       (df['STN'] == stn))[0]
    ssm_bare = df['WSM1'][stn_ind]
    ssm_bare = np.ma.masked_where(ssm_bare == -9999, ssm_bare)
    ssm_crop = df['WSM1.1'][stn_ind]
    ssm_crop = np.ma.masked_where(ssm_crop == -9999, ssm_crop)
    
    years = df['YEAR'][stn_ind]
    months = df['MN'][stn_ind]
    days = df['DT'][stn_ind]
    
    dates = np.zeros((len(years.values), 3))
    dates[:,0] = years.values
    dates[:,1] = months.values
    dates[:,2] = days.values
    
    date_inds = []
    
    for date in dates:
        date = map(int, date)
        try:
            date_inds.append(datetime(date[0], date[1], date[2]))
        except ValueError:
            date_inds.append(np.NAN)
    
    date_inds = np.array(date_inds)
    
    unmasked_bare = np.where(ssm_bare.mask == False)
    unmasked_crop = np.where(ssm_crop.mask == False)
    if plot == True:
        plt.figure()
        plt.plot(date_inds[unmasked_bare], ssm_bare[unmasked_bare], 'b')
        plt.plot(date_inds[unmasked_crop], ssm_crop[unmasked_crop], 'g')
        plt.plot(date_inds[unmasked_bare], ssm_bare[unmasked_bare], 'm.')
        plt.plot(date_inds[unmasked_crop], ssm_crop[unmasked_crop], 'k.')
        plt.legend(['ssm_bare', 'ssm_crop'])
        plt.title('IWMI surface soil moisture, station ' + stn)
        plt.show()
    
    df_bare = pd.DataFrame(ssm_bare.data[unmasked_bare],
                            index=date_inds[unmasked_bare])
    df_crop = pd.DataFrame(ssm_crop.data[unmasked_crop], 
                            index=date_inds[unmasked_crop])
    
    return df_bare, df_crop

    
def IWMI_ts_index(df, index, plot=False):

    stn_ind = np.where(df['INDEX'] == index)[0]
    ssm_bare = df['WSM1'][stn_ind]
    ssm_bare = np.ma.masked_where(ssm_bare == -9999, ssm_bare)
    ssm_crop = df['WSM1.1'][stn_ind]
    ssm_crop = np.ma.masked_where(ssm_crop == -9999, ssm_crop)
    
    years = df['YEAR'][stn_ind]
    months = df['MN'][stn_ind]
    days = df['DT'][stn_ind]
    
    dates = np.zeros((len(years.values), 3))
    dates[:,0] = years.values
    dates[:,1] = months.values
    dates[:,2] = days.values
    
    date_inds = []
    
    for date in dates:
        date = map(int, date)
        try:
            date_inds.append(datetime(date[0], date[1], date[2]))
        except ValueError:
            date_inds.append(np.NAN)
    
    date_inds = np.array(date_inds)
    
    unmasked_bare = np.where(ssm_bare.mask == False)
    unmasked_crop = np.where(ssm_crop.mask == False)
    if plot == True:
        plt.figure()
        plt.plot(date_inds[unmasked_bare], ssm_bare[unmasked_bare], 'b')
        plt.plot(date_inds[unmasked_crop], ssm_crop[unmasked_crop], 'g')
        plt.plot(date_inds[unmasked_bare], ssm_bare[unmasked_bare], 'm.')
        plt.plot(date_inds[unmasked_crop], ssm_crop[unmasked_crop], 'k.')
        plt.legend(['ssm_bare', 'ssm_crop'])
        plt.title('IWMI surface soil moisture, index ' + str(index))
        plt.show()
    
    df_bare = pd.DataFrame(ssm_bare.data[unmasked_bare],
                            index=date_inds[unmasked_bare])
    df_crop = pd.DataFrame(ssm_crop.data[unmasked_crop], 
                            index=date_inds[unmasked_crop])
    
    return df_bare, df_crop


if __name__ == "__main__":
    df = IWMI_read_csv()
    # mode can be 'index' or 'stn'
    ts_mode = 'stn'
    if ts_mode == 'stn':
        unique_stn = np.unique(np.array(df['STN']))
        stations = []
        for entry in unique_stn:
            stn = entry.split()
            stations.append(stn)
        stations = np.unique(np.array(stations))
        for stn in stations:
            print('start station ' + stn)
            try:
                IWMI_ts_stn(df, stn, plot=True)
            except IndexError:
                print('station ' + stn + ' does not work')
    
    elif ts_mode == 'index':
        station_idx = np.unique(np.array(df['INDEX']))
        
        for index in station_idx:
            print('start index ' + str(index))
            try:
                IWMI_ts_index(df, index, plot=True)
            except IndexError:
                print('index ' + str(index) + ' does not work')
    
    else:
        print('ts_mode must be either \'stn\' or \'index\'')
    
    print 'asdf'
