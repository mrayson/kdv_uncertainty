
# coding: utf-8

# # KdV Uncertainty output
# 
# In[1]:


import h5py
from glob import glob
import numpy as np
from datetime import datetime

import pandas as pd
import xarray as xr
from iwaves.utils import isw
import matplotlib.pyplot as plt

import dask
import os 
import sys
from time import gmtime, strftime, time

# Test manually setting the scheduler
dask.config.set(scheduler='processes')


################
# Inputs
#outpath = "slim-vi-lag-welbathy"
#outpath = 'slim-a0_optimal_GP'
#outpath = 'slim-a0_harmonic_GP'
#outpath = 'slim-harmonic_beta_a0'
#outpath = 'slim-harmonic_beta_pred_a0'
#outpath = 'slim-harmonic_beta_band6_a0'
outpath = 'slim-fblock-20072009'
#outpath = 'slim-harmonic_beta_band3_a0'
#nt = 367
nt = 1127
nsamples = 500
################

if "PYTHON_HOME" in os.environ.keys():
    PYTHON_HOME = os.environ["PYTHON_HOME"]
else:
    PYTHON_HOME = "."
sys.path.append(PYTHON_HOME)

slim_output_dir = os.path.join(PYTHON_HOME, "output", outpath)
slimfiles = os.path.join(slim_output_dir,'*.h5')
print(slimfiles)

timestamp = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
outfile = os.path.join(PYTHON_HOME,'output','{}_{}_magnus_kdv_runs.nc'.format(timestamp,outpath))


 
# Data are stored in individual hdf5 files for each step (1473) and each sample (500)


#def get_file_name(timepoint, sample):
#    return "../../kdv_solutions_azure/shiny_dashboard/data/timepoint-%d/all_amplitudes/2018-07-24--13-00-34_timepoint-%d_sample-%d_output.h5"%(timepoint,timepoint,sample)
#
#def get_all_file(timepoint):
#    return sorted(glob("../../kdv_solutions_azure/shiny_dashboard/data/timepoint-%d/all_amplitudes/*.h5"%(timepoint)))
#
#def get_summary_file(timepoint):
#    return sorted(glob("../../kdv_solutions_azure/shiny_dashboard/data/timepoint-%d/*.h5"%(timepoint)))[0]
##get_file_name(1,0)
##get_all_file(1)
##get_summary_file(1)

# In[4]:

# Get the summary file for new amplitudes
datafiles = sorted(glob(slimfiles))
if len(datafiles)==0:
    print('Found no data files!!!')

get_new_summary_file = {}
for df in datafiles:
    tstep = int(df.split('_')[-2].split('-')[-1])
    get_new_summary_file.update({tstep:df})
    

# # Data loading tools

def load_h5_step_slim(varname, timepoint):
    try:
        file = get_new_summary_file[timepoint]
        print(file)
    except:
        print('Cannot find file for step ', timepoint)
        return np.array([-999])

    h5 = h5py.File(file,'r')

    if varname not in list(h5.keys()):
        print('Cannot find variable %s. Returning -999.'%varname)
        a0 = np.array([-999])
    else:
        a0 = h5[varname].value


    h5.close()

    return a0

#amp, beta, a0 = load_h5_step(1)

#return_max_ensemble(1)

# # Load the newly generated data

# Get amax/a0 for all steps
#nt = 1479
amax_t = np.zeros((nsamples,nt))
tmax_t = np.zeros((nsamples,nt))
ubed_max_t = np.zeros((nsamples,nt))
usurf_max_t = np.zeros((nsamples,nt))

a0_t = np.zeros((nsamples,nt))
alpha_t = np.zeros((nsamples,nt))
alpha2_t = np.zeros((nsamples,nt))
cn_t = np.zeros((nsamples,nt))
beta_t = np.zeros((nt, nsamples,6))
alpha_mu_t = np.zeros((nsamples,nt))
alpha2_mu_t = np.zeros((nsamples,nt))
cn_mu_t = np.zeros((nsamples,nt))



# In[14]:


for ii in range(0,nt):

    a0_tmp = load_h5_step_slim('a0_samples', ii+1)
    amax_tmp = load_h5_step_slim('max_amplitude', ii+1)
    ubed_tmp = load_h5_step_slim('max_u_seabed', ii+1)
    usurf_tmp = load_h5_step_slim('max_u_surface', ii+1)
    beta_tmp = load_h5_step_slim('beta_samples', ii+1)

    ns = a0_tmp.shape[0]
    #if ns < 500:
    #    print('Only found %d sample for step %d'%(ns,ii))
    #alpha_tmp, c_tmp = calc_alpha(beta_tmp.T, zout, nsamples=ns, mode=0)    
    tmax_tmp = load_h5_step_slim('tmax', ii+1)
    c_tmp = load_h5_step_slim('c1', ii+1)
    r10 = load_h5_step_slim('r10', ii+1)
    alpha_tmp = -2*c_tmp*r10

    r20 = load_h5_step_slim('r20', ii+1)
    alpha2_tmp = -3*r20*c_tmp*c_tmp

    # Try getting the mean alpha and c variables
    #try:
    cmu_tmp = load_h5_step_slim('c1_mu', ii+1)
    r10_mu = load_h5_step_slim('r10_mu', ii+1)
    cn_mu_t[0:ns,ii] = cmu_tmp
    alpha_mu_t[:ns,ii] = -2*cmu_tmp*r10_mu

    r20_mu = load_h5_step_slim('r20_mu', ii+1)
    alpha2_mu_t[:ns,ii] = -3*cmu_tmp*cmu_tmp*r20_mu


    #except:
    #    print('No mean variables saved...')

        

    #print(a0_tmp.max())
    a0_t[0:ns,ii] = a0_tmp
    amax_t[0:ns,ii] = amax_tmp
    tmax_t[0:ns,ii] = tmax_tmp
    ubed_max_t[0:ns,ii] = ubed_tmp
    usurf_max_t[0:ns,ii] = usurf_tmp
    alpha_t[0:ns,ii] = alpha_tmp
    alpha2_t[0:ns,ii] = alpha2_tmp
    cn_t[0:ns,ii] = c_tmp

    beta_t[ii,0:ns,:] = beta_tmp.T


# In[15]:


# Create an xray dataset with the output
dims2 = ('ensemble','time')
dims3 = ('time','ensemble','params')

#time = rho.time.values
time = range(nt)
coords2 = {'time':time, 'ensemble':range(nsamples)}
coords3 = {'time':time, 'ensemble':range(nsamples), 'params':range(6)}
           
amax_da = xr.DataArray(amax_t,
                coords=coords2,
                dims=dims2,
                attrs={'long_name':'', 'units':''},
                )

tmax_da = xr.DataArray(tmax_t,
                coords=coords2,
                dims=dims2,
                attrs={'long_name':'', 'units':''},
                )

ubed_da = xr.DataArray(ubed_max_t,
                coords=coords2,
                dims=dims2,
                attrs={'long_name':'', 'units':''},
                )

usurf_da = xr.DataArray(usurf_max_t,
                coords=coords2,
                dims=dims2,
                attrs={'long_name':'', 'units':''},
                )
           
a0_da = xr.DataArray(a0_t,
    coords=coords2,
    dims=dims2,
    attrs={'long_name':'', 'units':''},
    )

cn_da = xr.DataArray(cn_t,
    coords=coords2,
    dims=dims2,
    attrs={'long_name':'', 'units':''},
    )

alpha_da = xr.DataArray(alpha_t,
    coords=coords2,
    dims=dims2,
    attrs={'long_name':'', 'units':''},
    )

alpha2_da = xr.DataArray(alpha2_t,
    coords=coords2,
    dims=dims2,
    attrs={'long_name':'', 'units':''},
    )



cn_mu_da = xr.DataArray(cn_mu_t,
    coords=coords2,
    dims=dims2,
    attrs={'long_name':'', 'units':''},
    )

alpha_mu_da = xr.DataArray(alpha_mu_t,
    coords=coords2,
    dims=dims2,
    attrs={'long_name':'', 'units':''},
    )

alpha2_mu_da = xr.DataArray(alpha2_mu_t,
    coords=coords2,
    dims=dims2,
    attrs={'long_name':'', 'units':''},
    )





beta_da = xr.DataArray(beta_t,
    coords=coords3,
    dims=dims3,
    attrs={'long_name':'', 'units':''},
    )

dsout = xr.Dataset({'amax':amax_da,\
    'tmax':tmax_da,\
    'a0':a0_da,\
    'cn':cn_da,\
    'alpha':alpha_da,\
    'alpha2':alpha2_da,\
    'cn_mu':cn_mu_da,\
    'cn_mu':cn_mu_da,\
    'alpha_mu':alpha_mu_da,\
    'alpha2_mu':alpha2_mu_da,\
    'beta':beta_da,\
    'ubed':ubed_da,\
    'usurf':usurf_da})

dsout


# In[16]:


dsout.to_netcdf(outfile)


