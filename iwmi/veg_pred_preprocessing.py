import os
import zipfile
import ast
import ConfigParser
from nc_stack_uptodate import check_stack, check_tiff_stack


def read_cfg(config_file):
    # read settings from config file
    configParser = ConfigParser.RawConfigParser()
    configFilePath = config_file
    configParser.read(configFilePath)
    cfg = {}
    for item, value in configParser.defaults().iteritems():
        cfg[item] = value

    return cfg


def unzip(path_in, path_out):
    '''
    Unzips folders from path_in to path_out

    Parameters:
    -----------
    path_in : str
        Path where zip files are stored
    path_out : str
        Path where all unzipped folders should be stored.

    '''
        
    folders = os.listdir(path_in)
    folders_uz = os.listdir(path_out)
    
    for fname in folders:
        if fname[4:12] in folders_uz:
            continue
        print fname
        zipfile_n = os.listdir(os.path.join(path_in, fname))[1]
        zip_ref = zipfile.ZipFile(os.path.join(path_in, fname, zipfile_n), 'r')
        zip_ref.extractall(path_out)
        zip_ref.close()


if __name__ == '__main__':
    
    # check and update SWI stack
    cfg = read_cfg('config_file_daily.cfg')
    
    swi_zippath = cfg['swi_zippath']
    data_path = cfg['swi_rawdata']
    unzip(swi_zippath, data_path)
    
    data_path_nc = cfg['swi_path_nc']
    nc_stack_path = cfg['swi_path']
    swi_stack_name = cfg['swi_stack_name']
    variables = cfg['swi_variables'].split()
    datestr = ast.literal_eval(cfg['swi_datestr'])
    
    check_stack(data_path, data_path_nc, nc_stack_path, swi_stack_name, 
                variables, datestr)
    
    # check and update VI stack
    data_path = cfg['vi_rawdata']
    data_path_nc = cfg['vi_path_nc']
    nc_stack_path = cfg['vi_path']
    swi_stack_name = cfg['vi_stack_name']
    variables = cfg['vi_variables']
    datestr = ast.literal_eval(cfg['vi_datestr'])
    
    check_tiff_stack(data_path, data_path_nc, nc_stack_path, swi_stack_name, 
                     variables, datestr)