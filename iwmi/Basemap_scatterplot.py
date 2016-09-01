import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as cls


def scatterplot(lons, lats, data, s=1, title=None, marker=','):
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(projection='cyl', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    
    parallels = np.arange(-90,90,15.)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = np.arange(-180,180,15.)
    m.drawmeridians(meridians,labels=[0,0,0,1])
    
    # define the colormap
    cmap = plt.get_cmap('jet', 20)
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    bounds = np.linspace(-120,120,241)
    norm = cls.BoundaryNorm(bounds, cmap.N)
    
    sc = m.scatter(lons, lats, c=data, edgecolor='None',
                   marker=marker, s=s, cmap=cmap, norm=norm)
    m.colorbar(sc, 'right', size='5%', pad='2%')
    if title:
        plt.title(title)
    plt.show()
    