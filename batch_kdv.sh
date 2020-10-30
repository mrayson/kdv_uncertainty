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
sbatch run-vkdv-singularity a0_samples_bandpass_6h_12month.h5 slim-harmonic_beta_band6_a0_bugfix ShellCrux_Filtered_Density_Harmonic_MCMC_201605_201705_prediction.h5 367
