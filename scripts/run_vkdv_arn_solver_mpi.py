#!/usr/bin/env python

"""
script to run the "IWaves" PDE solver on samples generated from the ddcurves2 model.
"""

import os
import sys

## define SOLITON_HOME, both for finding the iwaves package to import
## and to give absolute paths for output directories when running on Azure batch.
if "SOLITON_HOME" in os.environ.keys():
    SOLITON_HOME = os.environ["SOLITON_HOME"]
else:
    SOLITON_HOME = "."
sys.path.append(SOLITON_HOME)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from datetime import datetime
import xarray as xr
from scipy.interpolate import interp1d

from iwaves.kdv.vkdv import  vKdV
from iwaves.utils import imodes
from iwaves.utils import density

import h5py
#from multiprocessing import Pool, TimeoutError
from time import gmtime, strftime, time
import yaml

#from azure.storage.blob import BlockBlobService, PublicAccess

from mpi4py import MPI


# MPI variables
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#size = 1
#rank = 0

if rank == 0:
    print('MPI running: rank = %d, ncores = %d'%(rank,size))


def double_tanh(z, beta):
    res = beta[0] - beta[1] * (np.tanh((z + beta[2]) / beta[3]) + np.tanh((z + beta[4]) / beta[5]))
    return(res)

def doublesine(x, a_0, L_w, x0=0.):
    k = 2*np.pi/L_w
    k2 = 2*k
    eta =  - a_0 * np.cos(k*x + k*x0 + np.pi/2)
    eta +=  a_0/4. * np.cos(k2*x + k2*x0 + np.pi/2)
    eta[x>x0+3*L_w/2] = 0.
    #eta[x<x0-4*L_w/2] = 0.
    eta[x<x0-L_w/2] = 0.

    return eta

def zeroic(x,a0,Lw,x0=0):
    return 0.*x


# find the (signed) maximum deviation from zero

def maximum_amplitude_finder_old(amp_signal):
    amp_min = min(amp_signal)
    amp_max = max(amp_signal)
    #if abs(amp_min)>amp_max:
    #    return amp_min
    #else:
    #    return amp_max
    if np.abs(amp_min)>amp_max:
        return amp_min, np.argwhere(amp_signal==amp_min)[0][0]
    else:
        return amp_max, np.argwhere(amp_signal==amp_max)[0][0]
    
def maximum_amplitude_finder(amp_signal):
    amp_min = np.min(amp_signal)
    amp_max = np.max(amp_signal)
    if np.abs(amp_min)>amp_max:
        bidx = (amp_signal>=amp_min) & (amp_signal<= amp_min -0.01*amp_min)
        idx = np.argwhere(bidx)
        return amp_min, idx[0][0]
    else:
        bidx = (amp_signal<=amp_max) & (amp_signal>= amp_max -0.01*amp_max)
        idx = np.argwhere(bidx)
        return amp_max, idx[0][0]
    

def calc_u_velocity(kdv, A):
    # Linear streamfunction
    psi = A * kdv.phi_1 * kdv.c1
    # First-order nonlinear terms
    psi += A**2. * kdv.phi10 * kdv.c1**2.

    return np.gradient(psi, -kdv.dz_s)

def calc_u_velocity_1d(kdv, A, idx):
    # Linear streamfunction
    psi = A * kdv.Phi[:,idx] * kdv.c[idx]
    # First-order nonlinear terms
    #psi += A**2. * kdv.phi10[:,idx] * kdv.c[idx]**2.

    return np.gradient(psi, -kdv.dZ[idx])

# KdV functions
def bcfunc(F_a0, t, ramptime, twave=0, ampfac=1.):
    
    # Interpolate the boundary and apply the time offset and amplitude scaling
    # I think that I'm double counting the time-offset somewhere along the way
    a = F_a0(t-twave)/ampfac
    
    rampfac = 1 - np.exp(-(t)/ramptime)
    return a*rampfac

def start_kdv(infile, rho, z, depthfile):
    # Parse the yaml file
    with open(infile, 'r') as f:
        args = yaml.load(f,Loader=yaml.FullLoader)

        kdvargs = args['kdvargs']
        kdvargs.update({'verbose':False})
        #kdvargs.update({'nonlinear':False}) # Testing
        #kdvargs['Nsubset'] = 1


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

def init_vkdv_ar1( depthfile, infile, beta_ds, a0_ds, draw_num, t1, t2, basetime=datetime(2016,1,1)):
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

    rhonew = density.double_tanh_rho_new(z, *density_params)
    
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

def run_vkdv(F_a0, twave, ampfac, runtime, mykdv, infile, verbose=False, ramptime=12*3600.):
    
    # Need to reset the amplitude variables and time step
    mykdv.B *= 0 
    mykdv.B_n_m1 *= 0
    mykdv.B_n_m2 *= 0
    mykdv.B_n_p1 *= 0
    mykdv.t = 0 
    
    with open(infile, 'r') as f:
        args = yaml.load(f,Loader=yaml.FullLoader)

        kdvargs = args['kdvargs']
        kdvargs.update({'wavefunc':zeroic})

        #runtime = args['runtime']['runtime']
        ntout = args['runtime']['ntout']
        xpt =  args['runtime']['xpt']
        
    # Find the index of the output point
    idx = np.argwhere(mykdv.x > xpt)[0][0]

    # Initialise an output array
    nsteps = int(runtime//mykdv.dt)
    nout = int(runtime//ntout)
    B = np.zeros((nout, mykdv.Nx)) # Spatial amplitude function
    tout = np.zeros((nout,))

    B_pt = np.zeros((nsteps, )) # Spatial amplitude function
    tfast = np.zeros((nsteps,))

    def amp_at_x(kdv):
        #u_vel, w_vel = kdv.calc_velocity()
        u_vel = calc_u_velocity_1d(kdv, kdv.B[idx], idx)
        # u_vel is now a matrix size [Nx, Nz]
        u_surface = u_vel[0]
        u_seabed = u_vel[-1]

        return kdv.B[idx], u_surface, u_seabed


    output_amplitude = []
    

    ## Run the model
    nn=0
    for ii in range(nsteps):
        # Log output
        point = nsteps//100
        
        #bcleft = bcfunc(F_a0, twave, ampfac, mykdv.t, ramptime)
        bcleft = bcfunc(F_a0,  mykdv.t, ramptime, twave=-twave, ampfac=1.)
        #print(bcleft)
        
        if verbose:
            if(ii % (5 * point) == 0):
                print( '%3.1f %% complete...'%(float(ii)/nsteps*100)) 
                print(mykdv.B.max(), bcleft)

        if mykdv.solve_step(bc_left=bcleft) != 0:
            print( 'Blowing up at step: %d'%ii)
            break
        
        # Output data
        if (mykdv.t%ntout) < mykdv.dt:
            #print ii,nn, mykdv.t
            B[nn,:] = mykdv.B[:]
            tout[nn] = mykdv.t
            nn+=1

        # Output single point
        B_pt[ii] = mykdv.B[idx]
        tfast[ii] = mykdv.t


    max_output_amplitude, tidx = maximum_amplitude_finder(B_pt)
    tmax = tidx*mykdv.dt
    
    # Output the boundary amplitude
    a0 = bcfunc(F_a0, twave, ampfac, tfast, ramptime)
    max_a0,_ = maximum_amplitude_finder(a0)

    # Just output the last 24 hours of model solution. This ensures that the 
    # output vector length is the same for all simulations
    i0 = int(86400/mykdv.dt)

    return max_output_amplitude, max_a0,\
        B_pt[-i0:-1], tfast[-i0:-1],\
        tmax, mykdv, idx #, ds2.merge( ds )

def process_timepoint(timepoint, a0_ds, beta_ds, num_samples,
                     infile, depthfile, outpath,
                     ):
    """
    Process a single timepoint, doing num_samples samples.
    Save the output in an h5 file along with the input a0 and beta for this timepoint.
    If 'save_full' is True, also write out a file for each sample, containing all the amplitudes.
    If 'upload' is True, upload all outputs to cloud storage, using Azure credentials from env variables.
    """
    timestamp = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    slim_output_dir = os.path.join(SOLITON_HOME, "output",outpath)
    slim_outfile_name = os.path.join(slim_output_dir,\
        "{}_timepoint-{}_output.h5".format(timestamp,timepoint))


    t1 = beta_ds['time'].values[timepoint]
    t2 = beta_ds['time'].values[timepoint+1]

    all_amplitudes = []
    all_a0 = []
    all_c1 = []
    all_alpha = []
    all_beta = []
    all_c1_mu = []
    all_alpha_mu = []
    #all_r20 = []
    #all_r20_mu = []
    max_amplitudes = []
    max_u_surface_all = []
    max_u_seabed_all = []
    all_tmax = []
    all_density_params = []
    for sample in range(num_samples):
        if rank==0:
            print("Processing timepoint {}, sample {}, infile {}".\
                format(timepoint, sample, infile))

        tic = time()
        #max_amplitude, amplitudes, max_u_surface, max_u_seabed, tmax, mykdv, xpt =\
        #    run_solver(a0_sample, beta_sample, infile, depthfile,\
        #        rho_std, rho_mu, z_std)

        mykdv, F_a0, t0, runtime, density_params, twave, ampfac= \
            init_vkdv_ar1(depthfile, infile, beta_ds, a0_ds,\
                sample, t1, t2, basetime=datetime(2016,1,1))

        max_amplitude, a0, amplitudes, tfast, tmax, mykdv, xpt =\
            run_vkdv(F_a0, twave, ampfac, runtime, mykdv, infile, verbose=False)
        toc = time()

        if rank == 0:
            print("Amax %3.2f, Tmax %1.3e, run time: %3.2f seconds."%(
                max_amplitude, tmax, toc-tic))

       #     print("Ran solver for timepoint {}, sample {}".format(timepoint, sample))
        max_amplitudes.append(max_amplitude)
        #max_u_surface_all.append(max_u_surface)
        #max_u_seabed_all.append(max_u_seabed)
        all_amplitudes.append(amplitudes)
        all_a0.append(a0)

        #if max_amplitude != -999:
        all_c1.append(mykdv.c[xpt])
        all_alpha.append(mykdv.alpha[xpt])
        all_beta.append(mykdv.beta[xpt])
        all_tmax.append(tmax)
        all_density_params.append(density_params)

        if mykdv.ekdv:
            all_r20.append(mykdv.r20[xpt])

        # Calculate mean quantities\n",
        dx = mykdv.x[1]-mykdv.x[0]
        nx = mykdv.x.shape[0]
        L = np.arange(1,nx+1,1)*dx
        c_mu_t = np.cumsum(mykdv.c*dx) / L
        alpha_mu_t = np.cumsum(mykdv.alpha*dx) / L
        all_c1_mu.append(c_mu_t[xpt])
        all_alpha_mu.append(alpha_mu_t[xpt])
        if mykdv.ekdv:
            r20_mu_t = np.cumsum(mykdv.r20*dx) / L
            all_r20_mu.append(r20_mu_t[xpt])
        #else:
        #    # Run failed
        #    all_c1.append(-999)
        #    all_r10.append(-999)
        #    all_r01.append(-999)
        #    all_tmax.append(-999)
        #    all_c1_mu.append(-999)
        #    all_r10_mu.append(-999)



    #print(slim_outfile_name, density_params)
    timeout = (t0+tfast.astype('timedelta64[s]')).astype('<M8[ns]').astype(int)
    slim_outfile = h5py.File(slim_outfile_name,"w")
    #slim_outfile.create_dataset("time", data = timeout)
    slim_outfile.create_dataset("time", data = t1.astype('<M8[ns]').astype(int))
    slim_outfile.create_dataset("a0",data=np.array(all_a0))
    slim_outfile.create_dataset("A",data=np.array(all_amplitudes))
    slim_outfile.create_dataset("density_params",data=np.array(all_density_params))
    slim_outfile.create_dataset("max_amplitude",data=np.array(max_amplitudes))
    #slim_outfile.create_dataset("max_u_surface",data=np.array(max_u_surface_all))
    #slim_outfile.create_dataset("max_u_seabed",data=np.array(max_u_seabed_all))
    slim_outfile.create_dataset("c1",data=np.array(all_c1))
    slim_outfile.create_dataset("alpha",data=np.array(all_alpha))
    slim_outfile.create_dataset("c1_mu",data=np.array(all_c1_mu))
    slim_outfile.create_dataset("alpha_mu",data=np.array(all_alpha_mu))
    slim_outfile.create_dataset("beta",data=np.array(all_beta))
    slim_outfile.create_dataset("tmax",data=np.array(all_tmax))
    #
    #if mykdv != -1:
    #    if mykdv.ekdv:
    #        slim_outfile.create_dataset("r20_mu",data=np.array(all_r20_mu))
    #        slim_outfile.create_dataset("r20",data=np.array(all_r20))
    slim_outfile.close()

    return(None)

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the iwaves PDE solver")
    parser.add_argument("--beta_infile",default='./inputs/ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_prediction.h5')
    parser.add_argument("--a0_infile", default="./inputs/a0_samples_harmonicfit_M2S2N2K1O1_na0_AR5_12month.nc")
    parser.add_argument("--depthfile", default="./data/kdv_bathy_Prelude.csv")
    parser.add_argument("--infile", default="./data/kdvin.yml")
    parser.add_argument("--outpath", default="slim")
    parser.add_argument("--num_tp", help="number of timepoints to process in this job",
                        type=int,required=False)
    parser.add_argument("--num_samples", help="number of samples per timepoint",type=int,
                        required=False,default=500)
    #                    type=int,default=16)
    args = parser.parse_args()


    ### read in the beta and a0 samples here
    beta_ds = load_beta_h5(args.beta_infile)
    a0_ds = xr.open_dataset(args.a0_infile, group='predictions')

    ### what timepoints to do (linked to beta file)

    tp_min = 3 # Can't start from zero as we need some warm-up space
    if args.num_tp:
        tp_max = args.num_tp
    else:
        tp_max = beta_ds.time.shape[0]-1

    if rank == 0:
        print("Will execute timepoints {} to {} on core {}".format(tp_min, tp_max, rank))
        print('beta file {}'.format(args.beta_infile))
        print('a0 file {}'.format(args.a0_infile))
        print(72*'#')

        slim_output_dir = os.path.join(SOLITON_HOME, "output", args.outpath)
        try:
            os.mkdir(slim_output_dir)
        except:
            if rank == 0:
                print('Path {} exists'.format(slim_output_dir))

    comm.barrier()

    num_samples = args.num_samples
    if rank==0:
        print('# samples: {}'.format(num_samples))

    tps = range(tp_min, tp_max)
    numtime = len(tps)
    for ii in range(rank, numtime, size):
        tp = tps[ii]
        #try:
        process_timepoint(tp, a0_ds,
                 beta_ds, num_samples,
                 args.infile, args.depthfile, args.outpath,
        )
        #except:
        #    print('Timepoint {} failed! (rank={})'.format(ii,rank))
    
