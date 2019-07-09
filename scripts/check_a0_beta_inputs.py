"""
Check that the a0 file and beta file are compatible
"""

import xarray as xr
import numpy as np
import h5py
import matplotlib.pyplot as plt


###########33
a0file = 'inputs/a0_samples_harmonic_a0_all_times.h5'
betafile = '../run_ddcurves/DATA_SHELL/ShellCrux_Uniltered_Density_BHM_VI_20162017_nliw.nc'
L=1.5e5
###

ds = xr.open_dataset(betafile)

with h5py.File(a0file,'r') as f:
    a0 = f['data/a0-all-times-samples'][:]

nt,nsamples = a0.shape
nsamples=500

omega = 2*np.pi/(12.42*3600)

Ls = ds.cn[:,0:nsamples]**2 / (a0[:,0:nsamples]*omega*ds.alpha[:,0:nsamples])

plt.figure()
plt.hist(Ls.values.ravel()/1e5,bins=np.linspace(-2,2,100))
plt.hist(Ls.values.ravel()/1.5e5,bins=np.linspace(-2,2,100),color='r',alpha=0.3)
plt.xlim(-2,2)
plt.grid(b=True)
plt.show()

