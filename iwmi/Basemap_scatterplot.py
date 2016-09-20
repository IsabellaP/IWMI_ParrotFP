import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as cls
import matplotlib.colorbar as clb


def scatterplot(lons, lats, data, s=75, title=None, marker=',', discrete=True,
                **kwargs):
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(projection='cyl', ax=ax, llcrnrlat=10, urcrnrlat=35,
                    llcrnrlon=65, urcrnrlon=85)
    m.drawcoastlines()
    m.drawcountries()
    
    parallels = np.arange(-90,90,15.)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = np.arange(-180,180,15.)
    m.drawmeridians(meridians,labels=[0,0,0,1])
    
    m.readshapefile(os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                                 '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                                 'IND_adm2'), 'IN.MH.JN')
    
    if title:
        plt.title(title)
    
    if discrete:
        # define the colormap
        cmap = plt.get_cmap('jet', 20)
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        
        # define the bins and normalize
        bounds = np.linspace(kwargs['binmin'],kwargs['binmax'],kwargs['bins'])
        norm = cls.BoundaryNorm(bounds, cmap.N)
        
        sc = m.scatter(lons, lats, c=data, edgecolor='None',
                       marker=marker, s=s, cmap=cmap, norm=norm)
        ax2 = fig.add_axes([0.75, 0.1, 0.03, 0.8])
        clb.ColorbarBase(ax2, cmap=cmap, norm=norm, 
                         spacing='proportional', boundaries=bounds)
        ax2.set_yticklabels(kwargs['ticks'], minor=False)
    
    else:
        sc = m.scatter(lons, lats, c=data, edgecolor='None', marker=marker, 
                       s=s, vmin=kwargs['vmin'], vmax=kwargs['vmax'],
                       cmap='RdYlGn')
        m.colorbar(sc, 'right', size='5%', pad='2%')
        #plt.savefig('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\corr_'+kwargs['key']+'.png', bbox_inches='tight')
        
    plt.show()
    
    
def scatter_subplots(lons1, lats1, data1, s1,
                     lons2, lats2, data2, s2,
                     title1=None, title2=None):
    
    fig = plt.figure()
    
    # first subplot
    ax1 = fig.add_subplot(121)
    m = Basemap(projection='cyl', llcrnrlat=10, urcrnrlat=35,
                    llcrnrlon=65, urcrnrlon=85)
    m.drawcoastlines()
    m.drawcountries()
    
    parallels = np.arange(-90,90,15.)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = np.arange(-180,180,15.)
    m.drawmeridians(meridians,labels=[0,0,0,1])
    
    m.readshapefile(os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                                 '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                                 'IND_adm2'), 'IN.MH.JN')
    
    if title1:
        plt.title(title1)
        
    sc = m.scatter(lons1, lats1, c=data1, edgecolor='None', marker=',', 
                       s=s1, vmin=0, vmax=1, cmap='RdYlGn')
    m.colorbar(sc, 'right', size='5%', pad='2%')
    
    # second subplot
    ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
    m = Basemap(projection='cyl', llcrnrlat=10, urcrnrlat=35,
                    llcrnrlon=65, urcrnrlon=85)
    m.drawcoastlines()
    m.drawcountries()
    
    parallels = np.arange(-90,90,15.)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = np.arange(-180,180,15.)
    m.drawmeridians(meridians,labels=[0,0,0,1])
    
    m.readshapefile(os.path.join('C:\\', 'Users', 'i.pfeil', 'Documents', 
                                 '0_IWMI_DATASETS', 'shapefiles', 'IND_adm', 
                                 'IND_adm2'), 'IN.MH.JN')
    
    if title2:
        plt.title(title2)
        
    sc = m.scatter(lons2, lats2, c=data2, edgecolor='None', marker=',', 
                       s=s2, vmin=0, vmax=1, cmap='RdYlGn')
    m.colorbar(sc, 'right', size='5%', pad='2%')
    
    plt.show()
    