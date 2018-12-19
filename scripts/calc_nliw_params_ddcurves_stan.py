
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


def load_density_h5(h5file, samples):
    f = h5py.File(h5file,'r')
    data = f['data/beta_samples'][:,:,0:samples]
    #time = f['time'][:].astype('<M8[ns]')
    f.close()
    return data


def double_tanh(beta, z):
    
    return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
            + np.tanh((z+ beta[4])/beta[5]))

# Functions for reading input csv file data
def convert_time(tt):
    try:
        dt= datetime.strptime(tt, '%Y-%m-%dT%H:%M:%S')
    except:
        dt= datetime.strptime(tt, '%Y-%m-%d %H:%M')
    return dt

def read_density_csv(csvfile):
    # Reads into a dataframe object
    df = pd.read_csv(csvfile, index_col=0, sep=', ', parse_dates=['Time'], date_parser=convert_time)

    # Load the csv data
    depths= np.array([float(ii) for ii in df.columns.values])
    rho_obs_tmp = df[:].values.astype(float)
    time = df.index[:]

    # Clip the top
    rho_obs_2d = rho_obs_tmp[:,:]

    # Remove some nan
    fill_value = 1024.
    rho_obs_2d[np.isnan(rho_obs_2d)] = fill_value
    
    return xr.DataArray(rho_obs_2d,dims=('time', 'depth'),
            coords={'time':time.values,'depth':depths})

#########

def calc_nliw_params(infile, zmin, dz, mode=0, samples=500):
    
    h5file = "%s_beta-samples-array-all-data.h5"%infile
    csvfile = "%s.csv"%infile

    nparams = 6
    
    # Load the data
    data = load_density_h5(h5file,samples)

    # Load the input data from the csv file (this is currently not saved by the stan/r output)
    rho = read_density_csv(csvfile)
    time = rho.time.values


    nparam, nt, ntrace = data[:].shape
    
    zout = np.arange(0,zmin, -dz)

    
    # Calculate c and alpha
    #samples = ntrace
    alpha_ens = np.zeros((nt,samples))
    c_ens = np.zeros((nt,samples))

    rand_loc = np.random.randint(0, samples, samples)

    for tstep in tqdm(range(0,nt)):
        #if (tstep%20==0):
        #    print('%d of %d...'%(tstep,nt))
        for ii in range(samples):
            
            #elif nparams == 6:
            rhotmp = double_tanh(data[:,tstep, rand_loc[ii]], zout)

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
    coords2 = {'time':time, 'ensemble':range(samples)}
    coords3 = {'time':time, 'ensemble':range(samples), 'params':range(nparams)}


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

    dsout = xr.Dataset({'cn':cn_da, 'alpha':alpha_da, 'beta':beta_da,'rho':rho})
    print(dsout)
    
    return dsout


#############3
# Inputs

if __name__=='__main__':
    import sys
    infile = sys.argv[1]
    zmin = float(sys.argv[2])

    dz = 5.0

    outfile = '%s_nliw.nc'%infile

    print('%s\n Processing File %s '%(72*'#', infile))

    dsout = calc_nliw_params(infile, zmin, dz)

    dsout.to_netcdf(outfile)
    print('%s\n %s'%(outfile, 72*'#'))


