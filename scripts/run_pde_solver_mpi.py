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

from iwaves.kdv.kdv import  KdV
from iwaves.kdv.kdvimex import  KdVImEx as KdV
from iwaves.utils import density as density
from iwaves import IWaveModes
from iwaves import kdv, solve_kdv
from iwaves.utils.viewer import viewer

import h5py
#from multiprocessing import Pool, TimeoutError
from time import gmtime, strftime, time

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

def maximum_amplitude_finder(amp_signal):
        amp_min = min(amp_signal)
        amp_max = max(amp_signal)
        if abs(amp_min)>amp_max:
            return amp_min
        else:
            return amp_max

def calc_u_velocity(kdv, A):
    # Linear streamfunction
    psi = A * kdv.phi_1 * kdv.c1
    # First-order nonlinear terms
    psi += A**2. * kdv.phi10 * kdv.c1**2.

    return np.gradient(psi, -kdv.dz_s)



def run_solver(a0_sample, beta_sample):
    """
    instantiate an run the actual PDE solver, returning the full list of output amplitudes, plus the (signed)
    maximum amplitude.
    """
#    print("running kdv solver for a0 {}".format(a0_sample))
    dz = 2.5
    zmax = -252.5
    #z_new = np.linspace(zmax, 0, num = int(abs(zmax/dz)))
    z_new = np.arange(zmax,dz,dz)
#    print("p0 for a0 {}".format(a0_sample))
    profile_sample = double_tanh(z_new, beta_sample)
#    print("p0.5 for a0 {}".format(a0_sample))

    #iw_modes_output = IWaveModes(profile_sample,\
    #    z_new, density_class=density.InterpDensity)
    #    #z_new, density_class=density.FitDensity, density_func='double_tanh')
#   # print("p0.9 for a0 {}".format(a0_sample))
    #phi, c, he, zout = iw_modes_output(zmax, dz, 0)

#    print("p1 for a0 {}".format(a0_sample))
    # more runtime parameters
    dx = 50.
    L_d = 1.2e5
    eigen_mde = 0
    runtime = 1.5*86400.
    ntout = 1800.
    output_x = 75000.

    # if this outfile gets set to none, then the netcdf is not written to disk
    outfile = None

    kdvargs = dict(\
      verbose=False,\
      a0=a0_sample,\
      Lw=0.,\
      eigen_mde=eigen_mde,
      Cmax=0.8,\
      dt=20.,\
      nu_H=0.0,\
      ekdv=False,\
      wavefunc=zeroic,\
      spongedist = 10000.,\
      #L_d = L_d,\
      #Nx = int(np.ceil(2*L_d/dx)),\
      #Ricr=2.0,\
      #k_diss=1.0,
      )
#    print("p2 for a0 {}".format(a0_sample))
    omega = 2*np.pi/(12.42*3600)

    # No longer need this with zero initial conditions
    #k = omega/iw_modes_output.c1
    #Lw = 2*np.pi/k
    #kdvargs['Lw'] = Lw
    #kdvargs['x0'] = -1.5*Lw

    x_domain = np.arange(0,L_d+dx, dx)

    # Find the ouput x location
    xpt = np.argwhere(x_domain >= output_x)[0,0]

    
    # Boundary forcing function
    def sinebc(t):
        return a0_sample*np.sin(omega*t)


    # initalise the solver object
    #mykdv0 = kdv.KdV(iw_modes_output.rhoZ, iw_modes_output.Z, **kdvargs)
 #   print("initialized kdv object for a0 {}".format(a0_sample))

    # 8400 corresponds to the first grid point that is 1e5 meters away from
    # the initial point
    def amp_at_x(kdv):
      
      #u_vel, w_vel = kdv.calc_velocity()
      u_vel = calc_u_velocity(kdv, kdv.B[xpt])
      # u_vel is now a matrix size [Nx, Nz]
      u_surface = u_vel[0]
      u_seabed = u_vel[-1]

      return kdv.B[xpt], u_surface, u_seabed

    # Need seabed and seasurface currents
    # This is calculated using the calc_velocity method e.g.
    #    u_vel, w_vel = kdv.calc_velocity()

    # Maximum current profile?

    # Isotherm displacement amplitude

    # Wave length / Period?

    # Output in 2D space (x-y) would also be useful i.e. the projection of the maximum
    # seabed current onto an oblique structure like a pipeline. I have no idea
    # how to do these things

    #mykdv, B, output_amplitude = solve_kdv(iw_modes_output.rhoZ, iw_modes_output.Z, runtime,\
    mykdv, B, output_amplitude = solve_kdv(profile_sample, z_new, runtime,\
            solver='imex',\
            ntout=ntout, outfile=outfile,\
            x = x_domain,
            bcfunc=sinebc,
            myfunc=amp_at_x,\
            **kdvargs)
  #  print("finished running solver for a0 {}".format(a0_sample))
    #output_amplitude = np.array(output_amplitude[0])
    #output_u_surface = np.array(output_amplitude[1])
    #output_u_seabed = np.array(output_amplitude[2])
    output = np.array( [[aa[0], aa[1], aa[2]] for aa in output_amplitude])
    output_amplitude = output[:,0]
    output_u_surface = output[:,1]
    output_u_seabed = output[:,2]


    max_output_amplitude = maximum_amplitude_finder(output_amplitude)
    max_output_u_surface = maximum_amplitude_finder(output_u_surface)
    max_output_u_seabed = maximum_amplitude_finder(output_u_seabed)

  #  print("returning for a0 {}".format(a0_sample))
    return max_output_amplitude, output_amplitude, \
        max_output_u_surface, max_output_u_seabed



def process_timepoint(timepoint, a0_samples, beta_samples, num_samples,
                      save_all_amplitudes, upload):
    """
    Process a single timepoint, doing num_samples samples.
    Save the output in an h5 file along with the input a0 and beta for this timepoint.
    If 'save_full' is True, also write out a file for each sample, containing all the amplitudes.
    If 'upload' is True, upload all outputs to cloud storage, using Azure credentials from env variables.
    """
    timestamp = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    slim_output_dir = os.path.join(SOLITON_HOME, "output","slim")
    slim_outfile_name = os.path.join(slim_output_dir,\
        "{}_timepoint-{}_output.h5".format(timestamp,timepoint))
    slim_outfile = h5py.File(slim_outfile_name,"w")
    slim_outfile.create_dataset("timepoint", data = timepoint)
    slim_outfile.create_dataset("a0_samples",data=np.array(a0_samples))
    slim_outfile.create_dataset("beta_samples",data=np.array(beta_samples))
    all_amplitudes = []
    max_amplitudes = []
    max_u_surface_all = []
    max_u_seabed_all = []
    for sample in range(num_samples):
        if rank==0:
            print("Processing timepoint {}, sample {}".format(timepoint, sample))
        a0_sample = a0_samples[sample]
        beta_sample = beta_samples[:, sample]
        tic = time()
        max_amplitude, amplitudes, max_u_surface, max_u_seabed =\
            run_solver(a0_sample, beta_sample)
        toc = time()
        if rank == 0:
            print("run time: %3.2f seconds."%(toc-tic))
   #     print("Ran solver for timepoint {}, sample {}".format(timepoint, sample))
        max_amplitudes.append(max_amplitude)
        max_u_surface_all.append(max_u_surface)
        max_u_seabed_all.append(max_u_seabed)
        all_amplitudes.append(amplitudes)
        if save_all_amplitudes:
            full_output_dir = os.path.join(SOLITON_HOME, "output","full")
            full_outfile_name = os.path.join(full_output_dir,
                                             "{}_timepoint-{}_sample-{}_output.h5".format(timestamp,
                                                                                          timepoint,
                                                                                          sample))
            full_outfile = h5py.File(full_outfile_name,"w")
            full_outfile.create_dataset("timepoint", data=timepoint)
            full_outfile.create_dataset("sample", data=sample)
            full_outfile.create_dataset("a0_sample",data=a0_sample)
            full_outfile.create_dataset("beta_samples",data=beta_sample)
            full_outfile.create_dataset("amplitudes",data=amplitudes)
            full_outfile.create_dataset("max_amplitude",data=max_amplitude)
            full_outfile.close()
            if upload:
                blob_name = "timepoint-{}/all_amplitudes/{}_timepoint-{}_sample-{}_output.h5"\
                            .format(timepoint,timestamp,timepoint,sample)
                upload_file_to_azure(full_outfile_name,blob_name)
                pass
        pass
    slim_outfile.create_dataset("max_amplitude",data=np.array(max_amplitudes))
    slim_outfile.create_dataset("max_u_surface",data=np.array(max_u_surface_all))
    slim_outfile.create_dataset("max_u_seabed",data=np.array(max_u_seabed_all))
    slim_outfile.close()
    if upload:
        blob_name = "timepoint-" + str(timepoint)+"/" + timestamp + "_slim-output.h5"
        upload_file_to_azure(slim_outfile_name,blob_name)
        pass
    return(None)

#def upload_file_to_azure(file_path, blob_name):
#    """
#    upload a file to an azure blob.
#    Azure details (account name, key, container name) are obtained from env variables.
#    """
#    for var in ["AZURE_ACCOUNT","AZURE_KEY","AZURE_CONTAINER"]:
#        if not var in os.environ.keys():
#            print("{} environment variable not set".format(var))
#            return False
#    ### not really optimal to instantiate this service every time we want to upload a file,
#    ### but overhead shouldn't be noticeable, even if we upload the amplitudes every sample (i.e. every 20s or so).
#    block_blob_service = BlockBlobService(account_name=os.environ["AZURE_ACCOUNT"],
#                                          account_key=os.environ["AZURE_KEY"])
#    # check if the container exists.
#    if not block_blob_service.exists(os.environ["AZURE_CONTAINER"]):
#        block_blob_service.create_container(os.environ["AZURE_CONTAINER"])
#    block_blob_service.create_blob_from_path(os.environ["AZURE_CONTAINER"], blob_name, file_path)
#    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the iwaves PDE solver")
    parser.add_argument("--beta_infile",default='./inputs/2018-05-22_beta-samples-array-all-data.h5')
    parser.add_argument("--a0_infile", default="./inputs/2018-05-22_a0-samples-at-all-times.h5")
    parser.add_argument("--tp_min", help="start of timepoint range",type=int,required=False,default=0)
    parser.add_argument("--tp_max", help="end of timepoint range - if given",
                        type=int,required=False)
    parser.add_argument("--num_tp", help="number of timepoints to process in this job",
                        type=int,required=False)
    #parser.add_argument("--upload_to_azure", help="upload output to Azure",action='store_true')
    parser.add_argument("--save_all_amplitudes", help="save all output amplitudes",action='store_true')
    parser.add_argument("--num_samples", help="number of samples per timepoint",type=int,
                        required=False,default=500)
    parser.add_argument("--num_process", help="how many processes to run in parallel",
                        type=int,default=16)
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
        tp_max = 1473

    if rank == 0:
        print("Will execute timepoints {} to {} on core {}".format(tp_min, tp_max, rank))
### read in the beta and a0 samples here

    beta_file = h5py.File(args.beta_infile, 'r')
    beta_samples = np.array(beta_file['data/beta_samples'])

    a0_file = h5py.File(args.a0_infile, 'r')
    a0_samples = np.array(a0_file['data/a0-all-times-samples'])
    num_samples = args.num_samples
    #upload_to_azure = True if args.upload_to_azure else False
    upload_to_azure = False
    save_all_amplitudes = True if args.save_all_amplitudes else False

    tps = range(tp_min, tp_max)
    numtime = len(tps)
    for ii in range(rank, numtime, size):
        tp = tps[ii]
        process_timepoint(tp, a0_samples[tp,:num_samples],
                 beta_samples[:,tp,:num_samples],num_samples,
                 save_all_amplitudes, upload_to_azure)
    
