
# coding: utf-8

# # Find initial conditions for $$a_0$$ by optimizing the KdV model

# In[3]:


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import h5py

from scipy import signal
from scipy.optimize import minimize

from soda.utils.timeseries import timeseries, skill, rmse
from soda.utils.uspectra import uspectra, getTideFreq
from soda.utils.othertime import SecondsSince

#from iwaves.kdv.solve import solve_kdv
from iwaves.kdv.vkdv import  vKdV
import os


import matplotlib as mpl

import yaml



def double_tanh_6(beta, z):
    
    return beta[0,...] - beta[1,...]*(np.tanh((z+beta[2,...])/beta[3,...])
                + np.tanh((z+beta[2,...] + beta[4,...])/beta[5,...]))

def maximum_amplitude_finder(amp_signal):
    amp_min = np.min(amp_signal)
    amp_max = np.max(amp_signal)
    if np.abs(amp_min)>amp_max:
        return amp_min, np.argwhere(amp_signal==amp_min)[0][0]
    else:
        return amp_max, np.argwhere(amp_signal==amp_max)[0][0]

def zeroic(x, a_0, L_w, x0=0.):
    return 0*x

def start_kdv(infile, rho, z, depthfile):
    # Parse the yaml file
    with open(infile, 'r') as f:
        args = yaml.load(f)

        kdvargs = args['kdvargs']
        kdvargs.update({'wavefunc':zeroic})
        kdvargs.update({'verbose':False})

        runtime = args['runtime']['runtime']
        ntout = args['runtime']['ntout']
        xpt =  args['runtime']['xpt']


    # Parse the density and depth files
    depthtxt = np.loadtxt(depthfile, delimiter=',')

    # Initialise the KdV class
    mykdv = vKdV(rho,        z,        depthtxt[:,1],        x=depthtxt[:,0],        **kdvargs)

    return mykdv


def run_vkdv(a0,mykdv, infile, verbose=True):
    
    # Need to reset the amplitude variables and time step
    mykdv.B *= 0 
    mykdv.B_n_m1 *= 0
    mykdv.B_n_m2 *= 0
    mykdv.B_n_p1 *= 0
    mykdv.t = 0
    
    with open(infile, 'r') as f:
        args = yaml.load(f)

        kdvargs = args['kdvargs']
        kdvargs.update({'wavefunc':zeroic})

        runtime = args['runtime']['runtime']
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

    def bcfunc(t):
        omega = 2*np.pi/(12.42*3600.)
        return -a0*np.sin(omega*t)
        

    ## Run the model
    nn=0
    for ii in range(nsteps):
        # Log output
        point = nsteps//100
        if verbose:
            if(ii % (5 * point) == 0):
                 print( '%3.1f %% complete...'%(float(ii)/nsteps*100))
                 print(mykdv.B.max(), bcfunc(mykdv.t))

        if mykdv.solve_step(bc_left=bcfunc(mykdv.t)) != 0:
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
    attrs = {'long_name':'Wave amplitude',            'units':'m'}
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
    
    return maximum_amplitude_finder(B_pt)[0], ds.merge( ds2, )

def fdiff(a0, Amax, mykdv, infile):
    """
    Optimization function
    """

    Aguess,ds = run_vkdv(a0, mykdv, infile, verbose=False)
    
    # Change the sign Amax if the two aren't equal...
    if np.sign(Amax) != np.sign(Aguess):
        Amaxo = -1*Amax
    else:
        Amaxo = 1*Amax
        
    print(a0, Amaxo, Aguess )
    return (Aguess - Amaxo)**2.




def optimize_kdv(csvfile, depthfile, infile, outfilestr, overwrite=True):
    #pd.read_csv?

    # Load the csv file with the representative beta's and target A_max
    data = pd.read_csv(csvfile, sep=', ', parse_dates=['time','timemax'])

    # Load the depth
    depthtxt = np.loadtxt(depthfile, delimiter=',')
    z = np.arange(-depthtxt[0,1],5,5)[::-1]

    # For each time step
    for tt in data.index:
        print(72*'#')
        print('Optimizing for time step %s'%data['time'][tt])
        print(72*'#')


        outfile = '%s_%s.nc'%(outfilestr,data['time'][tt].strftime('%Y-%m-%d'))
        if os.path.exists(outfile):
            if overwrite is False:
                print('File %s exists moving on...'%outfile)
                continue


        # Load beta and Amax
        beta = np.array([data['beta0'][tt],\
            data['beta1'][tt],\
            data['beta2'][tt],\
            data['beta3'][tt],\
            data['beta4'][tt],\
            data['beta5'][tt]])
        Amax = data['Amax'][tt]
        rho = double_tanh_6(beta,z)


        # Launch a KdV instance
        mykdv =  start_kdv(infile, rho, z, depthfile)

        # Minimize
        print('Optimizing...')
        print('a_0, A_max, A_model')
        a0guess = min(np.abs(Amax),20)
        soln = minimize(fdiff, a0guess, 
                args = (Amax,mykdv, infile),
                bounds=[(0,50.0)],
                method='SLSQP', options={'eps':1e-4, 'ftol':1e-1}
                #method='CG',options={'eps':1e-4, 'gtol':1e-2}
                )

        print(soln)

        # Run the model again with the optimal solution and save the output
        Aguess, ds2 = run_vkdv( soln['x'], mykdv, infile, verbose=False)

        # Update the global attributes
        ds2.attrs.update({'Amax':Amax,'a0':soln['x'][0]})

        #plt.figure()
        #ds2.B_pt.plot()
        #
        #plt.figure()
        #ds2.B_t.plot()
        #ds2

        # Write to a file
        print('Saving to file: %s...'%outfile)
        ds2.to_netcdf(outfile)



####################################
# 

## Prelude input
#csvfile = 'data/vkdv_inputs_prelude.csv'
#depthfile = 'data/kdv_bathy_Prelude.csv'
#infile = 'data/kdvin_prelude_ekdv.yml'
#outfilestr = 'data/ekdv_optimal_a0_Prelude'
#optimize_kdv(csvfile, depthfile, infile, outfilestr)

# Prelude new bathy input
csvfile = 'data/vkdv_inputs_prelude_short.csv'
depthfile = 'data/kdv_bathy_Prelude_WELGA_bathy.csv'
infile = 'data/kdvin_prelude_kdv.yml'
outfilestr = 'data/kdv_optimal_a0_Prelude_NewBathy'
optimize_kdv(csvfile, depthfile, infile, outfilestr)


# IMOS PIL transect
#csvfile = 'data/vkdv_inputs_mode2_imospil200.csv'
#depthfile = 'data/kdv_bathy_PILIMOS_curved.csv'
#infile = 'data/kdvin_imospil_mode2_ekdv.yml'
#outfilestr = 'data/ekdv_mode2_optimal_a0_PILIMOS'
#overwrite=True
#optimize_kdv(csvfile, depthfile, infile, outfilestr, overwrite=overwrite)

#csvfile = 'data/vkdv_inputs_mode2_imospil200.csv'
#depthfile = 'data/kdv_bathy_PILIMOS_curved.csv'
#infile = 'data/kdvin_imospil_mode2_kdv.yml'
#outfilestr = 'data/kdv_mode2_optimal_a0_PILIMOS'
#overwrite=True
#optimize_kdv(csvfile, depthfile, infile, outfilestr, overwrite=overwrite)




#csvfile = 'data/vkdv_inputs_imospil200.csv'
#depthfile = 'data/kdv_bathy_PILIMOS_curved.csv'
#infile = 'data/kdvin_imospil_ekdv.yml'
#outfilestr = 'data/ekdv_optimal_a0_PILIMOS'
#overwrite=True
#optimize_kdv(csvfile, depthfile, infile, outfilestr, overwrite=overwrite)
#
#csvfile = 'data/vkdv_inputs_imospil200.csv'
#depthfile = 'data/kdv_bathy_PILIMOS_curved.csv'
#infile = 'data/kdvin_imospil.yml'
#outfilestr = 'data/kdv_optimal_a0_PILIMOS'
#optimize_kdv(csvfile, depthfile, infile, outfilestr, overwrite=overwrite)

###############


