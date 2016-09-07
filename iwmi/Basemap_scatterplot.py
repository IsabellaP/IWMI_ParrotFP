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
                       s=s, vmin=-0.6, vmax=1)
        m.colorbar(sc, 'right', size='5%', pad='2%')
        plt.savefig('C:\\Users\\i.pfeil\\Desktop\\poets\\RAWDATA\\corr_'+kwargs['key']+'.png', bbox_inches='tight')
        
    #plt.show()
    