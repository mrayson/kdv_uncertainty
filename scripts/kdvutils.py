"""
Utility routines to create kdv-like inputs
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from datetime import datetime
from iwaves.kdv.vkdv import  vKdV
from iwaves.utils import imodes
from iwaves.utils import density

import h5py
import yaml

# Input file parsers
def load_beta_h5(betafile):
    # Get the time from the beta file
    with h5py.File(betafile,'r') as f:
        t_beta=f['data/time'][:].astype('<M8[ns]')
        beta_samples = f['beta_samples'][:]
        z_std = np.array(f['data/z_std'])
        rho_std = np.array(f['data/rho_std'])
        rho_mu = np.array(f['data/rho_mu'])

    nparams, nt, nsamples = beta_samples.shape

    # Scale the beta parameters
    beta_samples[0,...] *= rho_std
    beta_samples[0,...] += rho_mu
    beta_samples[1,...] *= rho_std
    beta_samples[2::,...] *= z_std

    return xr.DataArray(beta_samples, dims=('params','time','draw'), 
                 coords={'time':t_beta,'params':range(nparams), 'draw':range(nsamples)})

# KdV functions
def bcfunc(F_a0, t, ramptime, twave=0, ampfac=1.):
    
    # Interpolate the boundary and apply the time offset and amplitude scaling
    a = F_a0(t-twave)/ampfac
    
    rampfac = 1 - np.exp(-(t)/ramptime)
    return a*rampfac

def start_kdv(infile, rho, z, depthfile):
    # Parse the yaml file
    with open(infile, 'r') as f:
        args = yaml.load(f,Loader=yaml.FullLoader)

        kdvargs = args['kdvargs']
        kdvargs.update({'verbose':False})
        runtime = args['runtime']['runtime']
        ntout = args['runtime']['ntout']
        xpt =  args['runtime']['xpt']


    # Parse the density and depth files
    depthtxt = np.loadtxt(depthfile, delimiter=',')
    N = depthtxt[:,0].shape[0]
    dx = depthtxt[1,0] - depthtxt[0,0]
    # Initialise the KdV class
    mykdv = vKdV(rho, z, depthtxt[:,1], depthtxt[:,0], N=N,
        dx=dx, **kdvargs)

    return mykdv


def myround(x, base=12*3600):
    return base * np.ceil(float(x)/base)

def init_vkdv_a0(depthfile, infile, beta_ds, a0_ds, draw_num, t1, t2, basetime=datetime(2016,1,1)):
    """
    Initialise the boundary conditions and the vKdV class for performing boundary condition
    inversion (optimization) calculations
    """
    # Find the observation location
    with open(infile, 'r') as f:
        args = yaml.load(f,Loader=yaml.FullLoader)
        xpt =  args['runtime']['xpt']
        Nz = args['runtime']['Nz']
    
    # Load the depth data
    depthtxt = np.loadtxt(depthfile, delimiter=',')
    #z = np.arange(-depthtxt[0,1],5,5)[::-1]
    z = np.linspace(-depthtxt[0,1],0,Nz)[::-1]

    
    # Load the density profile parameters
    density_params = beta_ds.sel(time=t1, draw=draw_num, method='nearest').values

    rhonew = density.double_tanh_rho_new2(z, *density_params)
    
    # Launch a KdV instance
    mykdv =  start_kdv(infile, rhonew, z, depthfile)

    # Find the index of the output point
    xpt = np.argwhere(mykdv.x > xpt)[0][0]
    
    # Compute the travel time and the wave amplification factor 
    ampfac = 1/np.sqrt(mykdv.Q)

    twave = np.cumsum(1/mykdv.c*mykdv.dx)
       
    # Set the time in the model to correspond with the phase of the boundary forcing
    ## Start time: round up to the near 12 hours from the wave propagation time plus the ramp time
    
    starttime = np.datetime64(t1)
    endtime = np.datetime64(t2)

    ramptime = 12*3600.
    bctime = np.timedelta64(int(myround(twave[xpt]+ramptime)),'s')

    runtime = (endtime - starttime).astype('timedelta64[s]').astype(float)

    t0 = starttime-bctime

    runtime = runtime+bctime.astype('timedelta64[s]').astype(float)

    # Need to return an interpolation object for a0 that will return a value for each model time step
    a0timesec = (a0_ds['time'].values-t0).astype('timedelta64[s]').astype(float)

    a0 =a0_ds['a0'].sel(draw=draw_num, chain=1).values

    F_a0 = interp1d(a0timesec, a0, kind=2)

    return mykdv, F_a0, t0, runtime, density_params, twave[xpt], ampfac[xpt]

