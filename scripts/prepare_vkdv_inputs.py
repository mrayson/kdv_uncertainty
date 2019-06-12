
# coding: utf-8

# # Create an input data set of $a_0$ and $\rho(z)$ for the vKdV model

# In[1]:


import h5py
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
#from mpl_toolkits.axisartist.
from matplotlib.collections import LineCollection

from glob import glob


# In[132]:


#%matplotlib notebook


# In[2]:


plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'axes.labelsize':'large'})


# In[3]:


def double_tanh_6(beta, z):
    
    return beta[0,...] - beta[1,...]*(np.tanh((z+beta[2,...])/beta[3,...])
                + np.tanh((z+beta[2,...] + beta[4,...])/beta[5,...]))


# In[118]:


def maximum_amplitude_finder(amp_signal):
    amp_min = np.min(amp_signal)
    amp_max = np.max(amp_signal)
    if np.abs(amp_min)>amp_max:
        return amp_min, np.argwhere(amp_signal==amp_min)[0][0]
    else:
        return amp_max, np.argwhere(amp_signal==amp_max)[0][0]


# In[37]:


def load_density_h5(h5file):
    f = h5py.File(h5file,'r')
    rho = f['rho'][:]
    depth = f['depth'][:]
    data = f['beta_samples'][:]
    time = f['time'][:].astype('<M8[ns]')
    f.close()
    return data,time, rho, depth

def load_density_xr(h5file):
    data,time, rho, depth = load_density_h5(h5file)

    nparams, nt, ntrace = data.shape
    dims3 = ('params','time','ensemble')

    coords3 = {'time':time, 'ensemble':range(ntrace), 'params':range(nparams)}



    beta_da = xr.DataArray(data,
        coords=coords3,
        dims=dims3,
        attrs={'long_name':'', 'units':''},
        )

    return xr.Dataset({'beta':beta_da,})


# In[64]:


###########
# Inputs
#datadir = '/home/suntans/cloudstor/Data/IMOS/'
datadir = '../run_ddcurves/DATA_SHELL/'

# ddcurves BHM output file
nparams=6
#h5file = '%s/Crux_KP150_12mth_Density_lowpass_density_bhm_6params_2018-12-16.h5'%datadir
h5file = '%s/Crux_KP150_Phs2_Density_lowpass_density_bhm_6params_2018-11-23.h5'%datadir

density_func = double_tanh_6


# In[72]:


# Load the output of ddcurves as an xarray object
ds = load_density_xr(h5file)
ds

plt.figure()
ds.beta[0,:,:].median(axis=-1).plot()
# In[73]:


# Load the amplitude and density data

ncfile = '/home/suntans/Share/ARCHub/DATA/FIELD/ShellCrux/KP150_Fitted_Buoyancy_wout_motion.nc'

ds1 = xr.open_dataset(ncfile,group='KP150_phs1')
ds2 = xr.open_dataset(ncfile,group='KP150_phs2')

# Merge the two
A_n = xr.concat([ds1['A_n'][:,0],ds2['A_n'][:,0]], dim='time')

plt.figure()
#ax=plt.subplot(211)
A_n.plot(lw=0.2)
plt.grid(b=True)
plt.ylim(-70,70)
# In[133]:


# Create a time series of single days with the max amplitude and a guess at beta
time1 = pd.date_range('2016-5-1','2016-9-15') 
time2 = pd.date_range('2016-11-1','2017-5-1')

#time = time1.append(time2)
time = time2


# In[ ]:


print('time, timemax, Amax, beta0, beta1, beta2, beta3, beta4, beta5')

for t1,t2 in zip(time[0:-1],time[1::]):
    A = A_n.sel(time=slice(t1,t2))
    
    # Find the amplitude
    Amax,idx = maximum_amplitude_finder(A.values)
    
    #plt.figure()
    #A.plot()
    #plt.plot(A.time.values[idx],Amax,'mo')
    
    # Find a representative (mean) beta
    dsc=ds.sel(time=t2,method='nearest')
    meanbeta = dsc.beta.median(axis=-1).values
    #print('{}, {}, {}'.format(t1, A.time.values[idx], Amax, ))
    outstr = '%s, %s, %3.2f'%(t1, A.time.values[idx], Amax, )
    for bb in meanbeta:
        outstr +=', %3.2f'%bb

    print(outstr)
    #print('{}, {}, {}, {}, {}, {}, {}, {}, {}'.format(t1, A.time.values[idx], Amax, dsc.beta.mean(axis=-1).values) )
    


