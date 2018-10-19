"""
Run (new) KdV code with Browse Basin KISSME data
"""

import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import h5py

import mycurrents.oceanmooring as om
from iwaves.utils.density import FitDensity, InterpDensity, double_tanh_rho_orig, single_tanh_rho
from iwaves import IWaveModes
from iwaves import kdv, solve_kdv 
from iwaves.utils.viewer import viewer

import pdb

def get_summary_file(timepoint):
    return sorted(glob("../kdv_solutions_azure/shiny_dashboard/data/timepoint-%d/*.h5"%(timepoint)))[0]

def return_a0_beta(timestep, ensemble):
    h5 = h5py.File(get_summary_file(timestep),'r')
    a0 = h5['a0_samples'][ensemble]
    beta = h5['beta_samples'][:,ensemble]
    h5.close()
    
    return a0, beta

def return_max_ensemble(timestep):
    h5 = h5py.File(get_summary_file(timestep),'r')
    a0 = h5['max_amplitude'][:]
    h5.close()

    return np.argwhere(np.abs(a0) == np.abs(a0).max())[0,0]

def double_tanh(beta, z):
    
    return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
                + np.tanh((z+beta[4])/beta[5]))

def fullsine(x, a_0, L_w, x0=0.):
    
    k = 2*np.pi/L_w
    eta =  - a_0 * np.cos(k*x + k*x0 + np.pi/2)
    eta[x>x0+3*L_w/2] = 0.
    #eta[x<x0-4*L_w/2] = 0.
    eta[x<x0-L_w/2] = 0.

    return eta

def doublesine(x, a_0, L_w, x0=0.):

    k = 2*np.pi/L_w
    k2 = 2*k
    eta =  - a_0 * np.cos(k*x + k*x0 + np.pi/2)
    eta +=  a_0/4. * np.cos(k2*x + k2*x0 + np.pi/2)
    eta[x>x0+3*L_w/2] = 0.
    #eta[x<x0-4*L_w/2] = 0.
    eta[x<x0-L_w/2] = 0.

    return eta

def run_kdv(timestep, ensemble, runtime):
    ########

    kdvargs = dict(\
        verbose=True,\
        a0=0,\
        Lw=5e4,\
        mode=0,
        #Cmax=0.8,\
        dt=20.,\
        nu_H=0.0,\
        ekdv=False,\
        wavefunc=doublesine,\
        L_d = 320000,
        Nx = 12800,
        alpha_10=0.0,\
        )

    ntout = 120.
    dz = 2.5

    #outfile = 'SCENARIOS/KISSME_KdV_test.nc'
    outfile = 'SCENARIOS/Browse_KdV_timestep%04d_ensemble%03d.nc'%\
        (timestep, ensemble)

    print( 72*'#')
    print( outfile)
    ########

    #T, salt, z = return_TS(ncfile, ncgroup, t1, t2)

    Z = np.arange(-250,dz,dz)[::-1]

    #rhoz = return_rhofit(paramfile, t1, Z, proftype=proftype)
    a0, beta = return_a0_beta(timestep, ensemble)
    rhoz= double_tanh(beta,Z)


    plt.plot(rhoz, Z)
    plt.show()

    ## Use the wrapper class to compute density on a constant grid
    #iw = IWaveModes(T, z, salt=salt, density_class=FitDensity, density_func=density_func)
    #iw2 = IWaveModes(T, z, salt=salt, density_class=InterpDensity)

    #phi, c1, he, Z = iw(-250,dz,mode)
    #phi2, c1a, hea, Z2 = iw2(-250,dz,mode)

    iw = kdv.KdVImEx(rhoz, Z, **kdvargs)
    iw.print_params()

    # Update the wavelength to represent an internal tide
    omega = 2*np.pi/(12.42*3600)
    k = omega/iw.c1
    Lw = 2*np.pi/k
    print( Lw)
    kdvargs['Lw'] = Lw
    kdvargs['a0'] = a0

    ### Test initialising the kdv class
    #mykdv0 = kdv.KdVImEx(iw.rhoZ, iw.Z, **kdvargs)
    #mykdv0.print_params()

    #print 'dx ', mykdv0.dx_s 

    ## Call the KdV run function
    mykdv, Bda = solve_kdv(rhoz, Z, runtime,\
            ntout=ntout, outfile=outfile, **kdvargs)

    return outfile

##############

timesteps = [13,300]
runtime = 1.5*86400.

for tt in timesteps:
    ee = return_max_ensemble(tt)
    outfile = run_kdv(tt, ee, runtime)

    # Call the viewer class directly
    V = viewer(outfile, tstep=-2, ulim=0.5, xaxis='distance')
    V.ax1.set_xlim(-30,0)
    V.ax2.set_xlim(-30,0)
    #
    plt.show()





