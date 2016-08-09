import os
import zipfile


def unzip(path_in, path_out):
        
    folders = os.listdir(path_in)
    
    for fname in folders:
        zipfile_n = os.listdir(os.path.join(path_in, fname))[1]
        zip_ref = zipfile.ZipFile(os.path.join(path_in, fname, zipfile_n), 'r')
        zip_ref.extractall(path_out)
        zip_ref.close()
        

if __name__ == '__main__':
    
    #path_in = 'C:\Users\s.hochstoger\Desktop\\0_IWMI_DATASETS\VIs\LAI_zipped'
    #path_out = 'C:\Users\s.hochstoger\Desktop\\0_IWMI_DATASETS\VIs\LAI'
    
    path_in = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\SWI_zipped\\'
    path_out = 'C:\\Users\\i.pfeil\\Documents\\0_IWMI_DATASETS\\SWI\\'
    
    unzip(path_in, path_out)