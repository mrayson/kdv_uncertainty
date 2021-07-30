"""
Generate real conditions for a KdV simulation

Use real depth and stratification data
"""

import numpy as np 
import scipy.io as io
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt
from datetime import datetime

from sfoda.dataio.conversion.dem import DEM
from sfoda.utils.myproj import MyProj


#######
# Inputs
basedir = '/home/suntans/cloudstor/Data/'
#depthfile = '%s/BATHYMETRY/ETOPO1/ETOPO1_Bed_TimorSea.nc'%basedir
#depthfile = '%s/BATHYMETRY/OTPS_Grid_Ind2016.nc'%basedir
#depthfile = '%s/BATHYMETRY/TimorSea_GA_GEBCO_Combined_DEM.nc'%basedir
depthfile = '%s/BATHYMETRY/TimorSea_GAWEL_Multi_GEBCO_Combined_DEM.nc'%basedir
#depthfile = '%s/BATHYMETRY/GEBCO_2014_TimorSea.nc'%basedir


#### Prelude transect (ending at Prelude)
#x0 = 122.840
#y0 = -13.080
#
#x1 = 123.3506
#y1 = -13.7641
#
## Prelude point
#xpt = 123.3506
#ypt = -13.7641


### Prelude transect
#x0 = 122.840
#y0 = -13.080
#
#x1 = 123.519
#y1 = -14.002
#

## WEL Bathy Prelude transect
x0 = 122.753
y0 = -13.1026

x1 = 123.486
y1 = -13.947
#x1 = 123.3506
#y1 = -13.7641

# Prelude point
xpt = 123.3506
ypt = -13.7641

### Rowley
#x0 = 119.005
#y0 = -17.796
#
#x1 = 120.005
#y1 = -18.938


## NRA Transet
#x0 = 115.829
#y0 = -19.269
#
#x1 = 116.418
#y1 = -20.057
#
#
#xpt = 116.
#ypt = -19.5

# IMOS PILBARA line
#x0 = 115.704
#y0 = -19.153
#
#x1 = 116.268
#y1 = -19.899
#xline = np.array( [115.589, 115.728, 115.819, 116.037, 116.285])
#yline = np.array( [-19.440, -19.303, -19.306, -19.562, -19.895])
#
#xpt = 115.914
#ypt = -19.436


dx = 250/1e5 # topo spacing (degrees)
# KdV parameters
#dxkdv = 50.
#spongedist=2e4
dxkdv = 1000.
spongedist=40000


#outfile_h = 'data/kdv_bathy_Prelude_coarse_5km.csv'
#outfile_h = 'data/kdv_bathy_Prelude_WELGA_bathy.csv'
outfile_h = 'data/kdv_bathy_Prelude_WELGA_bathy_1km.csv'
#######

## Generate x and y slice coordinates
lon = np.arange(x0, x1, dx)
Fx = interp1d([x0,x1],[y0,y1])
lat = Fx(lon)

#mydist = np.zeros_like(xline)
#mydist[1:] = np.cumsum(np.abs( (xline[1:]-xline[0:-1]) + 1j*(yline[1:]-yline[0:-1])))
#outdist = np.linspace(mydist[0],mydist[-1])
#
#kind = 'quadratic'
#Fx = interp1d(mydist,xline,kind=kind)
#lon = Fx(outdist)
#Fy = interp1d(mydist,yline,kind=kind)
#lat = Fy(outdist)
#
#plt.figure()
#plt.plot(lon,lat,'.')
#plt.plot(xline,yline,'ro')
#plt.show()


# Compute the distance coordinate
P = MyProj('merc')
x,y = P(lon,lat)
dx = np.diff(x)
dy = np.diff(y)
dist = np.zeros_like(x)
dist[1:] = np.cumsum(np.abs(dx + 1j*dy))

# Find the dist location of the nearest prelude location
distpt = np.abs( (xpt-lon) + 1j*(ypt-lat))
closest_idx = np.argwhere(distpt == distpt.min())[0][0]
print('Closest x-coordinate = %lf.'%(dist[closest_idx]))

#plt.plot(lon,lat)
#plt.plot(xpt,ypt,'x')
#plt.plot(lon[closest_idx],lat[closest_idx],'ro')
#plt.show()


# Load the DEM and interpolate
print( 'Interpolating the depth data...')
D = DEM(depthfile)
z = D.interp(lon,lat)
#
#plt.figure()
#plt.plot(dist, z)
#plt.show()

# Now create a domain for the vKdV class
#xkdv = np.arange(-xbuffer-2*Lw, dist[-1]+10*xbuffer+Lw, dxkdv) 
#xkdv = np.arange(-xbuffer-3*Lw, dist[-1]+10*xbuffer+Lw, dxkdv) 
#xkdv = np.arange(-0.1*spongedist, dist[-1]+2*spongedist, dxkdv) 
xkdv = np.arange(0, dist[-1]+spongedist, dxkdv) 

Fh = interp1d(dist, -z, bounds_error=False, fill_value=(-z[0],-z[-1]))
hkdv = Fh(xkdv)

# Interp the depth
Fh = interp1d(dist, -z, bounds_error=False, fill_value=(-z[0],-z[-1]))
hkdv = Fh(xkdv)
# Smooth the data
for ii in range(10):
    hkdv[10:-9] = np.convolve(hkdv,np.ones((20,))/20.,mode='valid')

topo_ds = np.array([xkdv, hkdv]).T


np.savetxt(outfile_h, topo_ds, delimiter=',', fmt='%3.6f')
print('Save files {} '.format(outfile_h))

plt.figure()
plt.plot(xkdv, -hkdv)
plt.grid(b=True)
plt.show()


