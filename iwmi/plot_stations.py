from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os


path = os.path.join('.', 'auxiliary', 'iwmi_warp_nn.csv')
dtype=np.dtype([('INDEX', np.int), ('STN', 'S4'),
                ('DE', np.int), ('MI', np.int),
                ('DE.1', np.int), ('MI.1', np.int), ('lat', np.float),
                ('lon', np.float), ('WARP_nn', np.int), ('Maps', 'S15'),
                ('STN_full_name', 'S15')])

data = np.genfromtxt(path, delimiter=',', dtype=dtype, skip_header=1)

# lons, lats stations IWMI
lon = data['lon']
lons = []
[lons.append(item) for item in lon if item not in lons]

lat = data['lat']
lats = []
[lats.append(item) for item in lat if item not in lats]

stations = data['STN']
ind = data['INDEX']
index = []
[index.append(item) for item in ind if item not in index]

# lons, lats COMoN Network
lons_c = [79.671, 78.5397, 76.834, 74.829, 76.3311, 92.7925]
lats_c = [29.6288, 17.4227, 17.3294, 14.1722, 10.0514, 26.6527]


# plot
map = Basemap(projection='merc', resolution = 'h',
              area_thresh = 0.1, llcrnrlon=65, llcrnrlat=6,
              urcrnrlon=100, urcrnrlat=34)

map.drawcoastlines(linewidth=0.5)
map.drawcountries(linewidth=0.5)
map.fillcontinents(color = 'coral', lake_color='aqua')
map.drawmapboundary(fill_color='aqua')

x,y = map(lons, lats)
map.plot(x, y, 'bo', markersize=10)
x_ismn, y_ismn = map(80.232889, 26.518883)
map.plot(x_ismn, y_ismn, 'ro', markersize=10)
x_c, y_c = map(lons_c, lats_c)
map.plot(x_c, y_c, 'go', markersize=10)

# Labels IWMI
labels_IWMI = index
x_offsets = [40000, 40000, 0, -420000, -420000, -420000, -420000, -420000, -420000, -420000]
y_offsets = [50000, 50000, -140000, -80000, -80000, -80000, -80000, -80000, -80000, -80000]
for label, xpt, ypt, x_offset, y_offset in zip(labels_IWMI, x, y, x_offsets, y_offsets):
    plt.text(xpt+x_offset, ypt+y_offset, label)

# Labels ISMN
plt.text(x_ismn-350000, y_ismn-80000, 'ISMN')

# Labels COMoN
labels_c = ['ALR', 'HYD', 'GBR', 'LGM', 'CHN', 'TZP']
x_offsets = [-150000, 10000, -130000, 50000, 10000, 40000]
y_offsets = [-140000, -120000, 80000, -30000, 50000, -20000]
for label, xpt, ypt, x_offset, y_offset in zip(labels_c, x_c, y_c, x_offsets, y_offsets):
    plt.text(xpt+x_offset, ypt+y_offset, label)

# legend
IWMI = mpatches.Patch(color='b', label='IWMI stations')
ISMN = mpatches.Patch(color='r', label='ISMN stations')
COMON = mpatches.Patch(color='g', label='COMoN stations')
plt.legend(handles=[IWMI, ISMN, COMON], loc=4)

#map.shadedrelief()

plt.show()