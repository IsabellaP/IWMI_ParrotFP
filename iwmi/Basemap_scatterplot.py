import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def scatterplot(lons, lats, data, s=1, title=None):
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(projection='cyl', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    
    parallels = np.arange(-90,90,15.)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = np.arange(-180,180,15.)
    m.drawmeridians(meridians,labels=[0,0,0,1])
    
    sc = m.scatter(lons, lats, c=data, edgecolor='None',
                   marker='o', s=s)
    m.colorbar(sc, 'right', size='5%', pad='2%')
    if title:
        plt.title(title)
    plt.show()