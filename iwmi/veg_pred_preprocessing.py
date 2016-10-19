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


if __name__ == '__main__':
    # check and update SWI stack
    cfg = read_cfg('config_file_daily.cfg')
    
    data_path = cfg['swi_rawdata']
    data_path_nc = cfg['swi_path_nc']
    nc_stack_path = cfg['swi_path']
    swi_stack_name = cfg['swi_stack_name']
    variables = cfg['swi_variables']
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