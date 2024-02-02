#####
# 4 trials
#    harmonic a0; data-restrained rho
#    harmonic a0; climatological rho
#    harmonic+stochastic a0; data-restrained rho
#    harmonic+stochastic a0; climatological rho
#
# Harmonics + stochastic component a0
#       a0_samples_harmonicfit_M2S2N2K1O1_na0_AR4_dt20min_12month.nc
## Deterministic harmonics only
#       a0_samples_harmonicfit_M2S2N2K1O1_na0_dt20min_12month.nc
# Data-restrained rho
#       ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_v5.h5
# climatologucal rho
#       ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_prediction_v5.h5
   
STOCHASTICA0=a0_samples_harmonicfit_M2S2N2K1O1_na0_AR4_dt20min_12month.nc
HARMONICA0=a0_samples_harmonicfit_M2S2N2K1O1_na0_dt20min_12month.nc
SEASONALA0=a0_samples_harmonicfit_M2S2N2K1O1_na3_dt60min_12month.nc
RHODATA=ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_v5.h5
RHOCLIM=ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_prediction_v5.h5
INFILE=./data/kdvin.yml
INFILEHYDRO=./data/kdvin_hydrostatic.yml

# Ocean Modelling Paper MC scenarios
sbatch run-vkdv-singularity $SEASONALA0 slim-seasonal_a0_data_rho_test $RHODATA 367 10 $INFILE
#sbatch run-vkdv-singularity $SEASONALA0 slim-seasonal_a0_data_rho_v4 $RHODATA 367 500 $INFILE
#sbatch run-vkdv-singularity $SEASONALA0 slim-seasonal_a0_clim_rho_v4 $RHOCLIM 367 500 $INFILE
#sbatch run-vkdv-singularity $HARMONICA0 slim-harmo_a0_data_rho_v4 $RHODATA 367 500 $INFILE
#sbatch run-vkdv-singularity $HARMONICA0 slim-harmo_a0_clim_rho_v4 $RHOCLIM 367 500 $INFILE

###
##sbatch run-vkdv-singularity $SEASONALA0 slim-seasonal_a0_data_rho_hydrostatic $RHODATA 367 500 $INFILEHYDRO
#sbatch run-vkdv-singularity $STOCHASTICA0 slim-stoch_a0_data_rho_v3 $RHODATA 367 500
#sbatch run-vkdv-singularity $STOCHASTICA0 slim-stoch_a0_clim_rho_v3 $RHOCLIM 367 500

#####
# Send a few vkdv runs to the queue
#sbatch run-vkdv-shifter a0_samples_harmonic_a0_variable_lag_2019-07-18.h5 slim-vi-lag-welbathy 
#sbatch run-vkdv-shifter a0_samples_optimal_a0_GP_all_times_2019-10-01.h5 slim-a0_optimal_GP
#sbatch run-vkdv-shifter a0_samples_harmonic_a0_GP_all_times_2019-09-27.h5 slim-a0_harmonic_GP
#sbatch run-vkdv-shifter a0_samples_harmonicfit_M2S2N2lowfreq_12month.h5 slim-harmonic_beta_a0 ShellCrux_Filtered_Density_Harmonic_MCMC_20162017.h5

#sbatch run-vkdv-shifter a0_samples_harmonicfit_M2S2N2lowfreq_12month.h5 slim-harmonic_beta_pred_a0 ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_prediction.h5
#sbatch run-vkdv-shifter a0_samples_harmonicfit_M2S2N2lowfreq_12month.h5 slim-harmonic_beta_pred_a0 ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_prediction.h5
#sbatch run-vkdv-shifter a0_samples_bandpass_3h_12month.h5 slim-harmonic_beta_band3_a0 ShellCrux_Filtered_Density_Harmonic_MCMC_201605_201705_prediction.h5 367
#sbatch run-vkdv-shifter a0_samples_bandpass_6h_12month.h5 slim-harmonic_beta_band6_a0 ShellCrux_Filtered_Density_Harmonic_MCMC_201605_201705_prediction.h5 367
#sbatch run-vkdv-shifter a0_samples_harmonicfit_FBlock_20072009.h5 slim-fblock-20072009 ShellCrux_Filtered_Density_Harmonic_MCMC_200709_200910_prediction.h5 1127
#sbatch run-vkdv-singularity a0_samples_bandpass_6h_12month.h5 slim-harmonic_beta_band6_a0_bugfix ShellCrux_Filtered_Density_Harmonic_MCMC_201605_201705_prediction.h5 367
#sbatch run-vkdv-singularity a0_samples_harmonicfit_M2S2N2lowfreq_12month.h5 slim-harmonic_beta_pred_a0_bugfix ShellCrux_Filtered_Density_Harmonic_MCMC_201605_201705_prediction.h5 367
#sbatch run-vkdv-singularity a0_samples_harmonicfit_M2S2nonstat_N2K1O1_12month.h5 slim-harmonic_beta_nonstat_a0_bugfix ShellCrux_Filtered_Density_Harmonic_MCMC_201605_201705_prediction.h5 367
####
#sbatch run-vkdv-singularity a0_samples_harmonicfit_M2S2N2K1O1_na0_AR5_12month.nc slim-AR_a0_harmonic_beta ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_prediction.h5 367 500
#sbatch run-vkdv-singularity a0_samples_harmonicfit_M2S2N2K1O1_na0_AR5_12month.nc slim-AR_a0_harmonic_beta ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_prediction.h5 24 5
