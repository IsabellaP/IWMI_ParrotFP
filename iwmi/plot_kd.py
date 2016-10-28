import os
import numpy as np
import matplotlib.pyplot as plt

path = 'C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\01_kd_param_20130531_JN\\'
kd = np.load(os.path.join(path, os.listdir(path)[0])).item()
keys = sorted(kd.keys())

plt.figure()

# kd for every pixel
for key in keys:
    for kd_file in os.listdir(path):
        kd = np.load(os.path.join(path, kd_file)).item()
        #key = (8,5)
        k = kd[key][0]
        d = kd[key][1]
        plt.plot(np.arange(100), np.arange(100)*k+d, "r")

    plt.title(str(key))
    plt.xlabel('SWI')
    plt.ylabel('D_VI')
    plt.savefig('C:\\Users\\i.pfeil\\Desktop\\veg_prediction\\10_plots\\'+str(key)+'.png')
    plt.close()
    
print 'done'
    