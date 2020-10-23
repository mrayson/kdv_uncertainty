"""
# Find initial conditions for $$a_0$$ by inverting the KdV model

"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import h5py

from scipy import signal
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from soda.utils.timeseries import timeseries, skill, rmse
from soda.utils.uspectra import uspectra, getTideFreq
from soda.utils.othertime import SecondsSince

#from iwaves.kdv.solve import solve_kdv
from iwaves.kdv.vkdv import  vKdV
from iwaves.utils import imodes
from iwaves.utils import density


import matplotlib as mpl

import yaml

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print('MPI running: rank = %d, ncores = %d'%(rank,size))
    verbose=True
else:
    verbose=False





###############
# KdV functions
###############
def zeroic(x, a_0, L_w, x0=0.):
    return 0*x

def bcfunc_old(t):
    omega = 2*np.pi/(12.42*3600.)
    return -a0*np.sin(omega*t)

def bcfunc(amp, frq, t, t0, ramptime):
    #omega = 2*np.pi/(12.42*3600.)
    nf = len(frq)
    a = 0.
    for nn in range(nf):
        a += amp[nn]*np.cos(frq[nn]*t) + amp[nn+nf]*np.sin(frq[nn]*t)
    
    rampfac = 1 - np.exp(-(t-t0)/ramptime)
    #print(t,t0, a, rampfac, a*rampfac)
    return a*rampfac


def start_kdv(infile, rho, z, depthfile):
    # Parse the yaml file
    with open(infile, 'r') as f:
        args = yaml.load(f)

        kdvargs = args['kdvargs']
        kdvargs.update({'wavefunc':zeroic})
        kdvargs.update({'verbose':False})
        #kdvargs.update({'nonlinear':False}) # Testing


        runtime = args['runtime']['runtime']
        ntout = args['runtime']['ntout']
        xpt =  args['runtime']['xpt']


    # Parse the density and depth files
    depthtxt = np.loadtxt(depthfile, delimiter=',')

    # Initialise the KdV class
    mykdv = vKdV(rho,\
        z,\
        depthtxt[:,1],\
        x=depthtxt[:,0],\
        **kdvargs)

    return mykdv


def run_vkdv(a0, frq, t0, runtime, mykdv, infile, verbose=True, ramptime=12*3600.):
    
    # Need to reset the amplitude variables and time step
    mykdv.B *= 0 
    mykdv.B_n_m1 *= 0
    mykdv.B_n_m2 *= 0
    mykdv.B_n_p1 *= 0
    mykdv.t = t0*1
    
    with open(infile, 'r') as f:
        args = yaml.load(f)

        kdvargs = args['kdvargs']
        kdvargs.update({'wavefunc':zeroic})

        #runtime = args['runtime']['runtime']
        ntout = args['runtime']['ntout']
        xpt =  args['runtime']['xpt']
        
    # Find the index of the output point
    idx = np.argwhere(mykdv.x > xpt)[0][0]

    # Initialise an output array
    nsteps = int(runtime//mykdv.dt_s)
    nout = int(runtime//ntout)
    B = np.zeros((nout, mykdv.Nx)) # Spatial amplitude function
    tout = np.zeros((nout,))

    B_pt = np.zeros((nsteps, )) # Spatial amplitude function
    tfast = np.zeros((nsteps,))

    output = []

    ## Run the model
    nn=0
    for ii in range(nsteps):
        # Log output
        point = nsteps//100
        bcleft = bcfunc(a0, frq, mykdv.t, t0, ramptime)
        
        if verbose:
            if(ii % (5 * point) == 0):
                print( '%3.1f %% complete...'%(float(ii)/nsteps*100)) 
                print(mykdv.B.max(), bcleft)

        if mykdv.solve_step(bc_left=bcleft) != 0:
            print( 'Blowing up at step: %d'%ii)
            break
        
        ## Evalute the function
        #if myfunc is not None:
        #    output.append(myfunc(mykdv))

        # Output data
        if (mykdv.t%ntout) < mykdv.dt_s:
            #print ii,nn, mykdv.t
            B[nn,:] = mykdv.B[:]
            tout[nn] = mykdv.t
            nn+=1

        # Output single point
        B_pt[ii] = mykdv.B[idx]
        tfast[ii] = mykdv.t

    # Save to netcdf
    ds = mykdv.to_Dataset()
    
    xray = xr

    # Create a dataArray from the stored data
    coords = {'x':mykdv.x, 'time':tout}
    attrs = {'long_name':'Wave amplitude',\
            'units':'m'}
    dims = ('time','x')

    Bda = xray.DataArray(B,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

    coords = {'timefast':tfast}
    attrs = {'long_name':'Wave Amplitude Point',
            'units':'m',
            'x-coord':xpt}
    dims = ('timefast',)
    Bpt = xray.DataArray(B_pt,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

    ds2 = xray.Dataset({'B_t':Bda,'B_pt':Bpt})
    #return ds2.merge( ds, inplace=True )
    #return ds.merge(ds2, inplace=True)
    #return ds.merge( xray.Dataset({'B_t':Bda,'B_pt':Bpt}), inplace=False )
    
    return ds2.merge( ds )

#############
# Initialisation functions
#############
def myround(x, base=12*3600):
    return base * np.ceil(float(x)/base)

def init_kdv_inversion(ds, depthfile, infile, t1, t2, mode, basetime=datetime(2016,1,1)):
    """
    Initialise the boundary conditions and the vKdV class for performing boundary condition
    inversion (optimization) calculations
    """
    
    # Get the time series of A(t)
    A_obs = ds['A_n'].sel(time=slice(t1,t2), modes=mode)
    
    # Get the density at the start of the time
    rho = ds['rhobar'].sel(timeslow=t1, method='nearest')
    
    # Load the depth data
    depthtxt = np.loadtxt(depthfile, delimiter=',')
    z = np.arange(-depthtxt[0,1],5,5)[::-1]
    
    # Get the density model parameters from the density profile
    iw = imodes.IWaveModes(rho.values[::-1], rho.z.values[::-1], \
            density_class=density.FitDensity, density_func='double_tanh')

    iw(-250,5,mode)

    density_params = iw.Fi.f0

    rhonew = density.double_tanh_rho(z, *density_params)
    
    
    # Launch a KdV instance
    mykdv =  start_kdv(infile, rhonew, z, depthfile)
    
    # Find the observation location
    with open(infile, 'r') as f:
        args = yaml.load(f)
        xpt =  args['runtime']['xpt']

    # Find the index of the output point
    xpt = np.argwhere(mykdv.x > xpt)[0][0]
    
    # Compute the travel time and the wave amplification factor 
    ampfac = 1/np.sqrt(mykdv.Qterm)

    twave = np.cumsum(1/mykdv.c1*mykdv.dx)
    
    # Compute the phase and amplitude of the signal
    At = timeseries(A_obs.time.values, A_obs.values)
    amp, phs, frq, _, Afit, _ = At.tidefit(frqnames=['M2','M4','M6'], basetime=basetime)
    
    # Now we need to scale the amplitude and phase for the boundary (linear inversion)
    phs_bc = 1*phs
    amp_bc = 1*amp
    for ii in range(3):
        phs_bc[ii] = phs[ii] - twave[xpt]*frq[ii]
        amp_bc[ii] = amp[ii] / ampfac[xpt]

    amp_re = amp_bc*np.cos(phs_bc)
    amp_im = amp_bc*np.sin(phs_bc)
    
    # Set the time in the model to correspond with the phase of the boundary forcing
    ## Start time: round up to the near 12 hours from the wave propagation time plus the ramp time


    ramptime = 6*3600.
    bctime = myround(twave[xpt]+ramptime)

    starttime = datetime.strptime(t1, '%Y-%m-%d %H:%M:%S')
    endtime = datetime.strptime(t2, '%Y-%m-%d %H:%M:%S')

    starttime_sec = (starttime - basetime).total_seconds() 

    runtime = (endtime - starttime).total_seconds() 

    #twave[xpt]+ramptime, bctime, runtime, starttime_sec
    
    # Testing only
    #ds2 = run_vkdv( np.hstack([amp_re,amp_im]), frq, starttime_sec-bctime, runtime+bctime, 
    #    mykdv, infile, verbose=False, ramptime=ramptime)
    
    # Input variables for the vKdV run
    # a0, frq, t0, runtime, 
    a0 = np.hstack([amp_re,amp_im])
    
    t0 = starttime_sec-bctime
    
    runtime = runtime+bctime
    
    return mykdv, At, a0, frq, t0, runtime, density_params, twave[xpt], ampfac[xpt]

########################
# Optimization routines
########################

def single_run(a0, frq, t0, runtime,  At, mykdv, infile):
    
    ds2 = run_vkdv(a0, frq, t0, runtime, mykdv, infile, verbose=False)
    
    tobs_sec = SecondsSince(At.t, basetime=basetime)
    F = interp1d(ds2.timefast, ds2.B_pt, bounds_error=False)
    Amod = F(tobs_sec)
    
    return ds2, Amod, tobs_sec
    
def fdiff(a0, frq, t0, runtime,  At, mykdv, infile,iter):

    ds2, Amod, tobs_sec = single_run(a0, frq, t0, runtime,  At, mykdv, infile)
    err = np.linalg.norm(At.y - Amod)
    
    iter+=1
    if verbose:
        print(iter, a0, err)
    
    # Return the L2-norm of the error vector
    return err

def print_result(xk):
    print(xk)
    

def invert_kdv(ds, depthfile, infile, t1, t2, mode, outpath, sitemname, basetime):
    
    if verbose:
        print(72*'#')
        print('Running inversion for {} to {}...'.format(t1,t2))

    # Extract the initial condition data and build the vKdV class
    print('\tCreating initial conditions...')
    mykdv, At, a0, frq, t0, runtime, density_params, twave, ampfac = \
        init_kdv_inversion(ds, depthfile, infile, t1, t2, mode, basetime=basetime)
    
    # Minimizeo
    iter=0
    print('\tMinimizing boundary condition parameters...')
    soln = minimize(fdiff, a0, 
            args = (frq, t0, runtime, At, mykdv, infile, iter),
            #callback=print_result,
            method='SLSQP', options={'eps':1e-1, 'ftol':1e-1, 'maxiter':50},
            #method='L-BFGS-B', options={'eps':1e-2, 'ftol':1e-2}    
            #method='Powell', options={'xtol':1e-2, 'ftol':1e-2}
            #method='CG',options={'eps':1e-4, 'gtol':1e-2}
            )

    # Re-run the solution with the best-fit
    print('\tRunning optimal solution...')
    ds2, Amod, t = single_run( soln['x'], frq, t0,  runtime, At, mykdv, infile)
    
    outfile = '{}/vkdv_bestfit_{}_{}.png'.format(outpath,sitename,t1[0:10])

    fig=plt.figure()
    plt.plot(At.t, At.y)
    plt.plot(At.t, Amod)
    plt.ylabel('A [m]')
    plt.legend(('Observed','vKdV'))
    plt.xticks(rotation=17)
    plt.ylim(-70,70)
    plt.xlim(At.t[0],At.t[-1])
    plt.grid(b=True,ls=':')
    plt.savefig(outfile)
    del fig
    
    # Save the optimized results and the important inputs in one netcdf file
    #soln['x'], a0, frq, At, Amod, t0, runtime, density_params, twave, ampfac, t1,t2,basetime

    h5file = '{}/vkdv_params_{}_{}.h5'.format(outpath,sitename,t1[0:10])
    if verbose:
        print('Wrote inputs to: ', h5file)

    with h5py.File(h5file, "w") as f:
        f['a0_init'] = a0
        f['a0_opt'] = soln['x']
        f['frq'] = frq
        f['A_obs'] = At.y
        f['A_mod'] = Amod
        f['density_params'] = density_params
        f['time_obs'] = At.t.astype(int)
        f['t0'] = t0
        f['t1'] = t1
        f['t2'] = t2
        f['twave'] = twave
        f['ampfac'] = ampfac
        f['basetime'] = datetime.strftime(basetime,'%Y-%m-%d %H:%M:%S')
        #for name in f:
        #    print(name)

    #ncfile = '{}/vkdv_soln_{}_{}.nc'.format(outpath,sitename,t1[0:10])
    #print('Wrote solution to : ',ncfile)
    #ds2.to_netcdf(ncfile)
    if verbose:
        print(72*'#')


if __name__=='__main__':
    #ncfile = '/home/suntans/Share/ARCHub/DATA/FIELD/ShellCrux/KP150_Fitted_Buoyancy_wout_motion_unvenfilt.nc'
    #ncpath = r'C:\Users\mrayson\cloudstor\Data\Crux'
    ncpath = '/home/mrayson/group/mrayson/DATA/FIELD/Crux'
    ncfile = '{}/KP150_Fitted_Buoyancy_wout_motion_unvenfilt.nc'.format(ncpath)
    depthfile = 'data/kdv_bathy_Prelude.csv'
    infile = 'data/kdvin.yml'

    sitename = 'CRUX_KP150'
    outpath = 'output/vkdv_optimization'

    # Load the amplitude and density time-series data
    mode = 0
    basetime = datetime(2016,1,1)

    ####
    # Batch 1
    ds = xr.open_dataset(ncfile, group='KP150_phs2')
    if verbose:
        print(ds.timeslow[0].values,ds.timeslow[-1].values)
    times = pd.date_range('2016-11-01','2017-05-08',freq='D')
    #for t1i,t2i in zip(times[0:-1],times[1::]):
    nsteps = times.shape[0]
    for ii in range(rank, nsteps-1, size): # Cycle through MPI
        t1,t2 = str(times[ii]), str(times[ii+1])
        invert_kdv(ds, depthfile, infile, t1, t2, mode, outpath, sitename, basetime)
    
    ####
    # Batch 2
    ds = xr.open_dataset(ncfile, group='KP150_phs1')
    if verbose:
        print(ds.timeslow[0].values,ds.timeslow[-1].values)
    times = pd.date_range('2016-05-01','2016-09-15',freq='D')
    #for t1i,t2i in zip(times[0:-1],times[1::]):
    nsteps = times.shape[0]
    for ii in range(rank, nsteps-1, size): # Cycle through MPI
        t1,t2 = str(times[ii]), str(times[ii+1])
        invert_kdv(ds, depthfile, infile, t1, t2, mode, outpath, sitename, basetime)
