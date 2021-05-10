"""
Bayesian estimation of $a_0$ 

This script uses a harmonic model only to represent $a_0$ i.e. no stochastic component
"""

import pymc3 as pm
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import h5py
from scipy.optimize import fmin_powell, fmin, fmin_cg, fmin_ncg

from theano import shared
from theano import tensor as tt
import arviz as az

from sfoda.utils.othertime import SecondsSince
from sfoda.utils.harmonic_analysis import getTideFreq
from mycurrents import oceanmooring as om

from scipy import signal
from tqdm import tqdm
import matplotlib as mpl


########
# Input variables

#a0file = '/home/suntans/Projects/ARCHub/DATA/FIELD/ShellCrux/KP150_Fitted_Buoyancy_wout_motion.nc'

basedir = '/home/suntans/cloudstor/Data/Crux/'
a0file ='{}/KP150_Fitted_Buoyancy_wout_motion_unvenfilt.nc'.format(basedir)


subsample = 60

tidecons = ['M2','S2','N2','K1','O1'] # Tidal harmonic
number_annual_harmonics = 3
basetime = datetime(2016,1,1)

sitename = 'M2S2N2K1O1_na{}_dt{}min'.format(number_annual_harmonics, subsample)
outputnc = '../inputs/a0_samples_harmonicfit_{}_12month.nc'.format(sitename)

# Prediction time
predtime = pd.date_range('2016-05-01','2017-05-02',freq='1H').values
########

##
# PyMC3 Bayesian Estimation Functions (can be moved elsewhere later...)

def sine_model_notrend(beta_s, ff, t, maths=pm.math):
    """
    Sum of sinusoids mean function
    
    Set maths == np to use numpy libary
    """
    n = len(ff)
    result = beta_s[0]+0*t
    for ii in range(0,n):
        result += beta_s[2*ii+1]*maths.cos(ff[ii] * t) + beta_s[2*ii+2]*maths.sin(ff[ii]*t)

    return result

def sine_model(beta_mu, beta_r, beta_i, ff, t, maths=pm.math):
    """
    Sum of sinusoids mean function
    
    Set maths == np to use numpy libary
    """
    n = len(ff)
    result = beta_mu+0*t
    for ii in range(0,n): 
        result += beta_r[ii]*maths.cos(ff[ii] * t) +\
            beta_i[ii]*maths.sin(ff[ii]*t)

    return result


def harmonic_fit_mcmc(time, X, frq, mask=None, axis=0, basetime=None,         **kwargs):
    """
    Harmonic fitting using Bayesian inference
    """
    tday = 86400.
    
    # Convert the time to days
    dtime = SecondsSince(time, basetime=basetime )
    dtime /= tday
    
    # Convert the frequencies to radians / day
    omega = [ff*tday for ff in frq]
    
    # Number of parameters
    n_params = 2*len(omega) + 1
    nomega = len(omega)
    
    print('Number of Parametrs: %d\n'%n_params, omega)

    with pm.Model() as my_model:
        ###
        # Create priors for each of our variables
        BoundNormal = pm.Bound(pm.Normal, lower=0.0)

        # Mean
        beta_mean = pm.Normal('beta_mean', mu=0, sd=1)
        
        
        beta_re = pm.Normal('beta_re', mu=1., sd = 5., shape=nomega)
        beta_im = pm.Normal('beta_im', mu=1., sd = 5., shape=nomega)

        #beta_s=[beta_mean]

        # Harmonics
        #for n in range(0,2*len(omega),2):
        #    beta_s.append(pm.Normal('beta_%d_re'%(n//2), mu=1., sd = 5.))
        #    beta_s.append(pm.Normal('beta_%d_im'%(n//2), mu=1., sd = 5.))
        
        # The mean function
        mu_x = sine_model(beta_mean, beta_re, beta_im, omega, dtime)
        
        ###
        # Generate the likelihood function using the deterministic variable as the mean
        sigma = pm.HalfNormal('sigma',5.)
        X_obs = pm.Normal('X_obs', mu=mu_x, sd=sigma, observed=X)
        
        mp = pm.find_MAP()
        print(mp)
        
        # Inference step...
        step = None
        start = None
        trace = pm.sample(500, tune=1000, start = start, step=step, cores=2,
                         )#nuts_kwargs=dict(target_accept=0.95, max_treedepth=16, k=0.5))
    
    # Return the trace and the parameter stats
    return trace,  my_model, omega, dtime 

def generate_harmonic_sample(dtime, omega, trace):
    true_center = 0.
    T = dtime.shape[0]
    y = np.zeros((T,))
    
    nomega = len(omega)
    
    #betas = [trace['beta_mean']]
    #for ii in range(nomega):
    #    betas.append(trace['beta_{}_re'.format(ii)])
    #    betas.append(trace['beta_{}_im'.format(ii)])
 
    # Add on an oscillatory component
    #mu = sine_model_notrend(betas, omega, dtime, maths=np)
    
    beta_mu = trace['beta_mean']
    beta_re = trace['beta_re']
    beta_im = trace['beta_im']
 
    # Add on an oscillatory component
    mu = sine_model(beta_mu, beta_re, beta_im, omega, dtime, maths=np)
    
    y+=mu
    
    return y


##################

###
# Load and pre-process the "observed/mode-fit" a0 data
# Preprocessing:
#  - Band-pass filter with a cut-off 3 - 34 hour
#  - Subsample the filtered data
###
print(72*'#')
print('Loading the input data...')

ds1 = xr.open_dataset(a0file,group='KP150_phs1')
ds2 = xr.open_dataset(a0file,group='KP150_phs2')# Filter each time series and concatenate


# Band-pass filter the 
A1 = om.OceanMooring(ds1.time.values, ds1['A_n'][:,0],0.0)
A2 = om.OceanMooring(ds2.time.values, ds2['A_n'][:,0],0.0)

A1f = om.OceanMooring(A1.t, A1.filt((34*3600, 3*3600), btype='band'), 0.0)
A2f = om.OceanMooring(A2.t, A2.filt((34*3600, 3*3600), btype='band'), 0.0)

A_n = A1f.concat(A2f)
A_n_1h = xr.DataArray(A_n.y[::subsample], dims=('time'), coords={'time':A_n.t[::subsample]})

####
# Create the inputs for the Bayesian estimation function

frq,names = getTideFreq(tidecons)

# Add on long-term modulations
tdaysec = 86400
fA = 2*np.pi/(365.25*tdaysec)

longperiods=[]
for n in range(-number_annual_harmonics, number_annual_harmonics+1):
    longperiods.append(n*fA)

frq_lt = []
for ff in frq:
    for ll in longperiods:
        frq_lt.append(ff+ll)

X_sd = 1
X_mu = 0

idx = ~np.isnan(A_n_1h.values)
X = A_n_1h.values[idx] - X_mu
X /= X_sd

timein = A_n_1h.time.values[idx]

print('Starting and building the Bayesian model...')

trace, my_model, omega, dtime = harmonic_fit_mcmc(timein, X, frq_lt,\
                                                              basetime=basetime)

print(pm.summary(trace))

######
# Generate samples
######
# Output the data at a new time
tsecnew = SecondsSince(predtime, basetime=basetime)
tdaynew = tsecnew/86400.

y_ar = []
for tt in trace:
    y_ar.append(generate_harmonic_sample(tdaynew, omega, tt))

a0_pred = np.array(y_ar)
    
#######
# Save the samples
#######
print('Saving the output to ', outputnc)

# Convert the data to arviz structure 
# Save the predictions
dims = ('chain','draw','time')
ds = az.from_pymc3_predictions({'a0':a0_pred}, \
                coords={'time':predtime,'chain':np.array([1])}, dims={'a0':dims})

# Save the posterior
ds2 = az.from_pymc3(trace=trace)

# Update the observed data becuase it comes out as a theano.tensor in the way
# our particular model is specified
ds2.observed_data['X_obs'] = xr.DataArray(X, dims=('time',), coords={'time':timein})

# This merges the data sets
ds2.extend(ds)

# Save 
ds2.to_netcdf(outputnc)

print(ds2)
print('Done')
print(72*'#')
