
# coding: utf-8

# # Plot output h5 data generated via ddcurves
# 
# 

# In[1]:


import h5py
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt

from iwaves import kdv
from iwaves.utils import isw
from tqdm import tqdm # progress bar
from glob import glob


def load_density_h5(h5file):
    f = h5py.File(h5file,'r')
    data = f['beta_samples'][:]
    time = f['time'][:].astype('<M8[ns]')
    f.close()
    return data,time

def double_tanh_6(beta, z):
    
    return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
            + np.tanh((z+beta[2] + beta[4])/beta[5]))

def double_tanh_7(beta, z):
    return beta[0] - beta[1]*np.tanh((z+beta[2])/beta[3]) \
        - beta[6]* np.tanh((z+beta[2] + beta[4])/beta[5])

def calc_nliw_params(h5file, zmin, dz, mode=0):
    
    # Lload the data
    data,time = load_density_h5(h5file)
    nparam, nt, ntrace = data[:].shape
    
    zout = np.arange(0,zmin, -dz)

    
    # Calculate c and alpha
    samples = ntrace
    alpha_ens = np.zeros((nt,samples))
    c_ens = np.zeros((nt,samples))

    rand_loc = np.random.randint(0, ntrace, samples)

    for tstep in tqdm(range(0,nt)):
        #if (tstep%20==0):
        #    print('%d of %d...'%(tstep,nt))
        for ii in range(samples):
            
            if nparams == 6:
                rhotmp = double_tanh_6(data[:,tstep, rand_loc[ii]], zout)
            elif nparams == 7:
                rhotmp = double_tanh_7(data[:,tstep, rand_loc[ii]], zout)


            N2 = -9.81/1000*np.gradient(rhotmp,-dz)

            phi,cn = isw.iwave_modes(N2, dz)

            phi_1 = phi[:,mode]
            phi_1 =phi_1 / np.abs(phi_1).max()
            phi_1 *= np.sign(phi_1.sum())

            alpha = isw.calc_alpha(phi_1, cn[mode],N2,dz)

            alpha_ens[tstep,ii] = alpha
            c_ens[tstep,ii] = cn[mode]
            #mykdv = kdv.KdV(rhotmp,zout)
            
    # Export to an xarray data set
    # Create an xray dataset with the output
    dims2 = ('time','ensemble',)
    dims3 = ('params','time','ensemble')

    #time = rho.time.values
    #time = range(nt)
    coords2 = {'time':time, 'ensemble':range(ntrace)}
    coords3 = {'time':time, 'ensemble':range(ntrace), 'params':range(nparams)}


    cn_da = xr.DataArray(c_ens,
        coords=coords2,
        dims=dims2,
        attrs={'long_name':'', 'units':''},
        )

    alpha_da = xr.DataArray(alpha_ens,
        coords=coords2,
        dims=dims2,
        attrs={'long_name':'', 'units':''},
        )

    beta_da = xr.DataArray(data,
        coords=coords3,
        dims=dims3,
        attrs={'long_name':'', 'units':''},
        )

    dsout = xr.Dataset({'cn':cn_da, 'alpha':alpha_da, 'beta':beta_da,})
    
    return dsout





#############3
# Inputs

datadir = '/home/suntans/cloudstor/Data/IMOS/'

sites = {
    'KP150':-252.5,
    'PIL200':-205,
    'KIM200':-200,
    'KIM400':-400,
    'ITFTIS':-460,
}
dz = 5.0


for sitename in sites.keys():
    for nparams in [6,7]:
        h5files = glob('%s/*_%s_*_%dparams_*.h5'%(datadir, sitename, nparams))
        for h5file in h5files:
            print('%s\n Processing File %s '%(72*'#', h5file))

            dsout = calc_nliw_params(h5file, sites[sitename], dz)

            outfile = '%s_nliw.nc'%h5file[:-14]
            dsout.to_netcdf(outfile)
            print('%s\n %s'%(outfile, 72*'#'))


