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

#from iwaves.kdv.kdv import  KdV
#from iwaves.kdv.kdvimex import  KdVImEx as KdV
from iwaves.kdv.vkdv import  vKdV
from iwaves.utils import density as density
from iwaves.utils.imodes import IWaveModes
#from iwaves.kdv.solve import solve_kdv
from iwaves.utils.viewer import viewer

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
    psi = A * kdv.phi_1[:,idx] * kdv.c1[idx]
    # First-order nonlinear terms
    psi += A**2. * kdv.phi10[:,idx] * kdv.c1[idx]**2.

    return np.gradient(psi, -kdv.dZ[idx])




def run_solver(a0_sample, beta_sample, infile, depthfile,\
        rho_std,rho_mu,z_std):
    """
    instantiate an run the actual PDE solver, returning the full list of output amplitudes, plus the (signed)
    maximum amplitude.
    """
   
    # Parse the yaml file
    with open(infile, 'r') as f:
        args = yaml.load(f)

        kdvargs = args['kdvargs']
        kdvargs.update({'wavefunc':zeroic})

        # Set the buoyancy eigenfunction to save time (not used here...)
        kdvargs.update({'D10':-1})
        kdvargs.update({'D01':-1})
        kdvargs.update({'D20':-1})

        runtime = args['runtime']['runtime']
        ntout = args['runtime']['ntout']
        output_x =  args['runtime']['xpt']

    # Parse the density and depth files
    depthtxt = np.loadtxt(depthfile, delimiter=',')

    x_domain = depthtxt[:,0]


    #    print("p2 for a0 {}".format(a0_sample))
    omega = 2*np.pi/(12.42*3600)

    #x_domain = np.arange(0,L_d+dx, dx)
    #    print("running kdv solver for a0 {}".format(a0_sample))
    #rho_std = 1.5
    #rho_mu = 1024.
    #z_std = 100.
    #z_new = np.arange(0, zmax,-dz)
    dz = 5
    z_new = np.arange(-depthtxt[0,1],dz,dz)[::-1]
    #rho_sample = double_tanh(z_new, beta_sample)
    rho_sample = double_tanh(z_new/z_std, beta_sample)*rho_std + rho_mu
 
    # Find the ouput x location
    xpt = np.argwhere(x_domain >= output_x)[0,0]
    
    # Boundary forcing function
    def bcfunc(t):
        return a0_sample*np.sin(omega*t)

    def amp_at_x(kdv):
        #u_vel, w_vel = kdv.calc_velocity()
        u_vel = calc_u_velocity_1d(kdv, kdv.B[xpt], xpt)
        # u_vel is now a matrix size [Nx, Nz]
        u_surface = u_vel[0]
        u_seabed = u_vel[-1]

        return kdv.B[xpt], u_surface, u_seabed

    # Parse the density and depth files
    depthtxt = np.loadtxt(depthfile, delimiter=',')

    # Initialise the KdV class
    try:
        mykdv = vKdV(rho_sample, z_new, depthtxt[:,1], \
            x=depthtxt[:,0], **kdvargs)
    except:
        print('Failed to init vKdV on core: ', rank)
        print('rho samples: ', rho_sample[0], rho_sample[-1], z_new[0],z_new[-1])
        print('betas: ', beta_sample)
        return -999, -1, -1, -1, -1, -1, -1



    ## Run the model
    try:
        nsteps = int(runtime // kdvargs['dt'])
        nn=0
        output_amplitude = []
        for ii in range(nsteps):
            if mykdv.solve_step(bc_left=bcfunc(mykdv.t)) != 0:
                print( 'Blowing up at step: %d'%ii)
                break
            
            # Evalute the function
            output_amplitude.append(amp_at_x(mykdv))
    except:
        print('Model crashed!!')
        print('rho samples: ', rho_sample[0], rho_sample[-1], z_new[0],z_new[-1])
        print('betas: ', beta_sample)
        return -999, -1, -1, -1, -1, -1, -1




    output = np.array( [[aa[0], aa[1], aa[2]] for aa in output_amplitude])
    output_amplitude = output[:,0]
    output_u_surface = output[:,1]
    output_u_seabed = output[:,2]

    #if rank == 0:
    #    plt.figure()
    #    plt.plot(output_u_seabed)
    #    plt.show()

    max_output_amplitude, tidx = maximum_amplitude_finder(output_amplitude)
    max_output_u_surface, _ = maximum_amplitude_finder(output_u_surface)
    max_output_u_seabed, _ = maximum_amplitude_finder(output_u_seabed)
    tmax = tidx*mykdv.dt_s

    return max_output_amplitude, output_amplitude, \
        max_output_u_surface, max_output_u_seabed, tmax, mykdv, xpt



def process_timepoint(timepoint, a0_samples, beta_samples, num_samples,
                     infile, depthfile, outpath,
                     rho_std, rho_mu, z_std,
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
    slim_outfile = h5py.File(slim_outfile_name,"w")
    slim_outfile.create_dataset("timepoint", data = timepoint)
    slim_outfile.create_dataset("a0_samples",data=np.array(a0_samples))
    slim_outfile.create_dataset("beta_samples",data=np.array(beta_samples))
    all_amplitudes = []
    all_c1 = []
    all_r10 = []
    all_r01 = []
    all_c1_mu = []
    all_r10_mu = []
    all_r20 = []
    all_r20_mu = []
    max_amplitudes = []
    max_u_surface_all = []
    max_u_seabed_all = []
    all_tmax = []
    for sample in range(num_samples):
        if rank==0:
            print("Processing timepoint {}, sample {}, a0 {}, infile {}".\
                format(timepoint, sample, a0_samples[sample], infile))
        a0_sample = a0_samples[sample]
        beta_sample = beta_samples[:, sample]
        tic = time()
        max_amplitude, amplitudes, max_u_surface, max_u_seabed, tmax, mykdv, xpt =\
            run_solver(a0_sample, beta_sample, infile, depthfile,\
                rho_std, rho_mu, z_std)
        toc = time()

        if rank == 0:
            print("Amax %3.2f, Tmax %1.3e, run time: %3.2f seconds."%(
                max_amplitude, tmax, toc-tic))

       #     print("Ran solver for timepoint {}, sample {}".format(timepoint, sample))
        max_amplitudes.append(max_amplitude)
        max_u_surface_all.append(max_u_surface)
        max_u_seabed_all.append(max_u_seabed)
        all_amplitudes.append(amplitudes)

        if max_amplitude != -999:
            all_c1.append(mykdv.c1[xpt])
            all_r10.append(mykdv.r10[xpt])
            all_r01.append(mykdv.r01[xpt])
            all_tmax.append(tmax)

            if mykdv.ekdv:
                all_r20.append(mykdv.r20[xpt])

            # Calculate mean quantities\n",
            dx = mykdv.x[1]-mykdv.x[0]
            nx = mykdv.x.shape[0]
            L = np.arange(1,nx+1,1)*dx
            c_mu_t = np.cumsum(mykdv.c1*dx) / L
            r10_mu_t = np.cumsum(mykdv.r10*dx) / L
            all_c1_mu.append(c_mu_t[xpt])
            all_r10_mu.append(r10_mu_t[xpt])
            if mykdv.ekdv:
                r20_mu_t = np.cumsum(mykdv.r20*dx) / L
                all_r20_mu.append(r20_mu_t[xpt])
        else:
            # Run failed
            all_c1.append(-999)
            all_r10.append(-999)
            all_r01.append(-999)
            all_tmax.append(-999)
            all_c1_mu.append(-999)
            all_r10_mu.append(-999)

    slim_outfile.create_dataset("max_amplitude",data=np.array(max_amplitudes))
    slim_outfile.create_dataset("max_u_surface",data=np.array(max_u_surface_all))
    slim_outfile.create_dataset("max_u_seabed",data=np.array(max_u_seabed_all))
    slim_outfile.create_dataset("c1",data=np.array(all_c1))
    slim_outfile.create_dataset("r10",data=np.array(all_r10))
    slim_outfile.create_dataset("c1_mu",data=np.array(all_c1_mu))
    slim_outfile.create_dataset("r10_mu",data=np.array(all_r10_mu))
    slim_outfile.create_dataset("r01",data=np.array(all_r01))
    slim_outfile.create_dataset("tmax",data=np.array(all_tmax))
    
    if mykdv != -1:
        if mykdv.ekdv:
            slim_outfile.create_dataset("r20_mu",data=np.array(all_r20_mu))
            slim_outfile.create_dataset("r20",data=np.array(all_r20))
    slim_outfile.close()

    return(None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the iwaves PDE solver")
    parser.add_argument("--beta_infile",default='./inputs/ShellCrux_Uniltered_Density_BHM_VI_20162017.h5')
    parser.add_argument("--a0_infile", default="./inputs/a0_samples_harmonic_a0_all_times.h5")
    parser.add_argument("--depthfile", default="./data/kdv_bathy_Prelude.csv")
    parser.add_argument("--infile", default="./data/kdvin_prelude.yml")
    parser.add_argument("--outpath", default="slim")
    parser.add_argument("--tp_min", help="start of timepoint range",type=int,required=False,default=0)
    parser.add_argument("--tp_max", help="end of timepoint range - if given",
                        type=int,required=False)
    parser.add_argument("--num_tp", help="number of timepoints to process in this job",
                        type=int,required=False)
    parser.add_argument("--num_samples", help="number of samples per timepoint",type=int,
                        required=False,default=500)
    #                    type=int,default=16)
    args = parser.parse_args()

    ### what timepoints to do:
    if args.tp_max and args.num_tp:
        raise RuntimeError("Error - only one of tp_max and num_tp should be set")
    tp_min = args.tp_min
    if args.num_tp:
        tp_max = tp_min + args.num_tp
    elif args.tp_max:
        tp_max = args.tp_max
    else:
        tp_max = 1480

    if rank == 0:
        print("Will execute timepoints {} to {} on core {}".format(tp_min, tp_max, rank))
        print('beta file {}'.format(args.beta_infile))
        print('a0 file {}'.format(args.a0_infile))
        print(72*'#')

        slim_output_dir = os.path.join(SOLITON_HOME, "output", args.outpath)
        os.mkdir(slim_output_dir)

    comm.barrier()

    ### read in the beta and a0 samples here

    beta_file = h5py.File(args.beta_infile, 'r')
    beta_samples = np.array(beta_file['beta_samples'])
    z_std = np.array(beta_file['data/z_std'])
    rho_std = np.array(beta_file['data/rho_std'])
    rho_mu = np.array(beta_file['data/rho_mu'])

    if rank == 0:
        print('rho_mu {}, rho_std: {}, z_std: {}'.format(rho_mu,rho_std,z_std))

    a0_file = h5py.File(args.a0_infile, 'r')
    a0_samples = np.array(a0_file['data/a0-all-times-samples'])
    num_samples = args.num_samples

    tps = range(tp_min, tp_max)
    numtime = len(tps)
    for ii in range(rank, numtime, size):
        tp = tps[ii]
        process_timepoint(tp, a0_samples[tp,:num_samples],
                 beta_samples[:,tp,:num_samples],num_samples,
                 args.infile, args.depthfile, args.outpath,
                 rho_std, rho_mu, z_std,
             )
    
