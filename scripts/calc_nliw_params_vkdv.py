
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
import os

import pdb


def load_density_h5(h5file):
    f = h5py.File(h5file,'r')
    rho = f['data/rho'][:]
    depth = f['data/z'][:]
    data = f['beta_samples'][:]
    time = f['data/time'][:].astype('<M8[ns]')
    rho_std = f['data/rho_std'].value
    rho_mu = f['data/rho_mu'].value
    z_std = f['data/z_std'].value
    f.close()
    return data,time, rho, depth, rho_std, z_std, rho_mu

def single_tanh(beta, z):
    
    return beta[0] - beta[1]*np.tanh((z+beta[2])/beta[3])

def double_tanh(beta, z):
    
    return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
            + np.tanh((z+beta[4])/beta[5]))

def double_tanh_6(beta, z):
    
    return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
            + np.tanh((z+beta[2] + beta[4])/beta[5]))

def double_tanh_7(beta, z):
    return beta[0] - beta[1]*np.tanh((z+beta[2])/beta[3]) \
        - beta[6]* np.tanh((z+beta[2] + beta[4])/beta[5])

def calc_nliw_params(h5file, depthtxt, dz, mode=0, samples=None):
    
    # Lload the data
    data,time, rho, depth, rho_std, z_std, rho_mu = load_density_h5(h5file)
    nparams, nt, ntrace = data[:].shape
    
    
    # Calculate c and alpha
    if samples is None:
        samples = ntrace

    alpha_ens = np.zeros((nt,samples))
    c_ens = np.zeros((nt,samples))
    alpha_mu_ens = np.zeros((nt,samples))
    c_mu_ens = np.zeros((nt,samples))

    # Depth file paramaters
    x = depthtxt[:,0]
    zmins = -depthtxt[:,1]
    L = x[-1]
    dx = x[1]-x[0]
    nx = x.shape[0]


    rand_loc = np.random.randint(0, ntrace, samples)
    beta_out = np.zeros((nparams, nt, samples))

    for tstep in  tqdm(range(0,nt)):
        #if (tstep%20==0):
        #    print('%d of %d...'%(tstep,nt))

        print(tstep, nt, nx)
        for ii in tqdm(range(samples)):

            alpha_mu = 0
            cn_mu = 0
            for kk in range(nx):
                zmin = zmins[kk]
                zout = np.arange(0,zmin, -dz)
            
                if nparams == 4:
                    rhotmp = single_tanh(data[:,tstep, rand_loc[ii]], zout/z_std)
                elif nparams == 6:
                    rhotmp = double_tanh(data[:,tstep, rand_loc[ii]], zout/z_std)
                elif nparams == 7:
                    rhotmp = double_tanh_7(data[:,tstep, rand_loc[ii]], zout/z_std)

                beta_out[:,tstep, ii] = data[:,tstep, rand_loc[ii]]

                # Need to scale rho

                rhotmp = rhotmp*rho_std + rho_mu

                N2 = -9.81/1000*np.gradient(rhotmp,-dz)

                phi,cn = isw.iwave_modes(N2, dz)

                phi_1 = phi[:,mode]
                phi_1 =phi_1 / np.abs(phi_1).max()
                phi_1 *= np.sign(phi_1.sum())

                alpha = isw.calc_alpha(phi_1, cn[mode],N2,dz)

                alpha_mu += alpha*dx
                cn_mu += cn[mode]*dx

                # Always leave the last value as alpha and cn
                alpha_ens[tstep,ii] = alpha
                c_ens[tstep,ii] = cn[mode]

            alpha_mu_ens[tstep,ii] = alpha_mu/L
            c_mu_ens[tstep,ii] = cn_mu/L
 
            #mykdv = kdv.KdV(rhotmp,zout)
            
    # Export to an xarray data set
    # Create an xray dataset with the output
    dims2 = ('time','ensemble',)
    #dims2a = ('time','depth',)
    dims3 = ('params','time','ensemble')

    #time = rho.time.values
    #time = range(nt)
    coords2 = {'time':time, 'ensemble':range(samples)}
    #coords2a = {'time':time, 'depth':depth[:,0]}
    coords3 = {'time':time, 'ensemble':range(samples), 'params':range(nparams)}


    #rho = xr.DataArray(rho.T,
    #    coords=coords2a,
    #    dims=dims2a,
    #    attrs={'long_name':'', 'units':''},
    #    )
     
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

    cn_mu_da = xr.DataArray(c_mu_ens,
        coords=coords2,
        dims=dims2,
        attrs={'long_name':'', 'units':''},
        )

    alpha_mu_da = xr.DataArray(alpha_mu_ens,
        coords=coords2,
        dims=dims2,
        attrs={'long_name':'', 'units':''},
        )



    beta_da = xr.DataArray(beta_out,
        coords=coords3,
        dims=dims3,
        attrs={'long_name':'', 'units':''},
        )

    dsout = xr.Dataset({'cn':cn_da, 'alpha':alpha_da, 'beta':beta_da,\
        'cn_mu':cn_mu_da, 'alpha_mu':alpha_mu_da,})
    
    return dsout





#############3
# Inputs

#datadir = '/home/suntans/cloudstor/Data/IMOS/'
datadir = '../run_ddcurves/DATA_SHELL/'

dz = 5.


h5files =[ 
    #'%s/ShellCrux_Unfiltered_Density_BHM_MCMC_0203Aug2016.h5'%datadir,
    #'%s/ShellCrux_Unfiltered_Density_BHM_VI_0203Aug2016.h5'%datadir,
    #'%s/ShellCrux_Filtered_Density_BHM_MCMC_20162017.h5'%datadir,
    #'%s/ShellCrux_Filtered_Density_BHM_MCMC_Jul2016.h5'%datadir,
    #'%s/ShellCrux_Filtered_Density_BHM_VI_20162017.h5'%datadir,
    #'%s/ShellCrux_Filtered_Density_BHM_VI_Jul2016.h5'%datadir,
    #'%s/ShellCrux_Uniltered_Density_BHM_MCMC_20162017.h5'%datadir,
    '%s/ShellCrux_Uniltered_Density_BHM_VI_20162017.h5'%datadir,
    ]

depthfile = 'data/kdv_bathy_Prelude_coarse_5km.csv'

depthtxt = np.loadtxt(depthfile, delimiter=',')


for h5file in h5files:
    outfile = '%s_vkdv_nliw.nc'%h5file[:-3]
    dsout = calc_nliw_params(h5file, depthtxt, dz, samples=500)
    dsout.to_netcdf(outfile)
    print('%s\n %s'%(outfile, 72*'#'))




