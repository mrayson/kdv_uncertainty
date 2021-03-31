"""
Estimate the parameters of a parametric density curve with a seasonal mean function
"""

import numpy as np
import pandas as pd

import pymc3 as pm

from pyddcurves import utils
from pyddcurves import models as dd

###########
# Input variables

## Constants for all cases
z_std = 100.
rho_std = 1.5
rho_mu = 1024.

omega_A = 2*np.pi/365.

omega = [omega_A, 2*omega_A, 3*omega_A, 4*omega_A]
#omega = [omega_A, 2*omega_A, 3*omega_A]

tune = 2000 # No. of MCMC tuning steps

basedir = './data'
datafile1 = '%s/Shell_Density_KP150_phs1'%basedir
datafile2 = '%s/Shell_Density_KP150_phs2'%basedir

outdir = 'inputs'
outfile = '%s/ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_v5.h5'%outdir
outfile_pred = '%s/ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_prediction_v5.h5'%outdir

# Prediction time
kdvtime = pd.date_range('2016-05-01','2017-05-02',freq='D').values

### Time span for the first two deployments
kdvtime1 = pd.date_range('2016-05-01','2016-10-31',freq='D').values
kdvtime2 = pd.date_range('2016-11-01','2017-05-02',freq='D').values
###########


#####
# Create an observation data dictionary
#####

print('Loading the input data...')
rho1 = utils.read_density_csv('%s.csv'%datafile1)
rho2 = utils.read_density_csv('%s.csv'%datafile2)

# Interpolate on a constant time grid
rho1i = rho1.interp({'time':kdvtime1}, method='nearest',kwargs={"fill_value": "extrapolate"})
rho2i = rho2.interp({'time':kdvtime2}, method='nearest',kwargs={"fill_value": "extrapolate"})

ntimeavg = 1 

rho = rho1i
nt,nz = rho.shape

depths_2d = rho.depth.values[np.newaxis,:].repeat(nt, axis=0)

obsdata1 = utils.density_to_obsdict(rho.values, depths_2d, rho.time.values, ntimeavg, z_std, rho_mu, rho_std)

rho = rho2i
nt,nz = rho.shape

depths_2d = rho.depth.values[np.newaxis,:].repeat(nt, axis=0)

obsdata2 = utils.density_to_obsdict(rho.values, depths_2d, rho.time.values, ntimeavg, z_std, rho_mu, rho_std)

obsdata = utils.merge_obs_dicts(obsdata1, obsdata2)

####
# Perform inference on the data
####

print('Starting and building the Bayesian model...')
trace, model, tdays = dd.density_bhm_harmonic_dht(obsdata, omega, use_mcmc=True, nchains=2, ncores=2, tune=tune)

print(pm.summary(trace))

print('Saving to file {}...'.format(outfile))
# Save the result
utils.bhm_harmonic_to_h5(outfile, trace, obsdata, omega)

####
# Predict using the mean (climatological) level of the hierarchy
print('Saving mean data to file {}...'.format(outfile_pred))
new_beta_samples = utils.beta_prediction(outfile, kdvtime, outfile=outfile_pred, scaled=False)
