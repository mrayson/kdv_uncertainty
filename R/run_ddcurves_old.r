library(ddcurves2)
library(rstan)
library(hdf5r)

stan_dat_one <- list(
  n_depths = length(Crux_KP150_Phs1$depths),
  n_times = nrow(Crux_KP150_Phs1$density_mat),
  depths = Crux_KP150_Phs1$depths,
  densities = Crux_KP150_Phs1$density_mat
)

stan_dat_two <- list(
  n_depths = length(Crux_KP150_Phs2$depths),
  n_times = nrow(Crux_KP150_Phs2$density_mat),
  depths = Crux_KP150_Phs2$depths,
  densities = Crux_KP150_Phs2$density_mat
)

#Some initial values are required (Although I hope to remove this step in
#the future)

n_t_one <- stan_dat_one$n_times
init_sub <- list(
  beta_zero = rep(1025, n_t_one),
  beta_one = rep(1, n_t_one),
  beta_three = rep(42, n_t_one),
  beta_six = rep(48, n_t_one),
  beta_midpoint = array(c(rep(77, n_t_one), rep(150, n_t_one)), dim = c(n_t_one, 2)),
  mean_beta_zero = 1025,
  mean_beta_one = 1,
  mean_beta_three = 42,
  mean_beta_six = 48,
  mean_beta_midpoint = array(c(77, 150), dim = 2),
  sigma_beta = array(c(0.18, 0.12, 26.1, 8.09, 12.6, 6.65), dim = 6),
  sigma_curve = 0.08
)

n_t_two <- stan_dat_two$n_times
init_sub <- list(
  beta_zero = rep(1025, n_t_two),
  beta_one = rep(1, n_t_two),
  beta_three = rep(42, n_t_two),
  beta_six = rep(48, n_t_two),
  beta_midpoint = array(c(rep(77, n_t_two), rep(150, n_t_two)), dim = c(n_t_two, 2)),
  mean_beta_zero = 1025,
  mean_beta_one = 1,
  mean_beta_three = 42,
  mean_beta_six = 48,
  mean_beta_midpoint = array(c(77, 150), dim = 2),
  sigma_beta = array(c(0.18, 0.12, 26.1, 8.09, 12.6, 6.65), dim = 6),
  sigma_curve = 0.08
)

#The density profile models are then fit:

model_fit_one <- double_tanh_no_timeseries(
  stan_data_list = stan_dat_one,
  iter = 1750,
  warmup = 1000,
  save_warmup = FALSE,
  chains = 2,
  cores = 2,
  refresh = 5,
  control = list(adapt_delta = 0.9, max_treedepth = 16),
  init = list(init_sub, init_sub)
)

model_fit_two <- double_tanh_no_timeseries(
  stan_data_list = stan_dat_two,
  iter = 1750,
  warmup = 1000,
  save_warmup = FALSE,
  chains = 2,
  cores = 2,
  refresh = 5,
  control = list(adapt_delta = 0.9, max_treedepth = 16),
  init = list(init_sub, init_sub)
)

### Combining the model fits, and writing to disk

#Now we have to combine the samples generated from the model fits into
#one structure. We also slightly reorder the samples here, as the
#`double_tanh` function expects them in a certain order.
#
#First, we get the relevant parameter names, and extract the samples of
#said parameters
#
#``` r
par_names <- grep("^beta", model_fit_one@model_pars, value = TRUE)

first_half_beta_samples <- extract(model_fit_one, par_names)
first_half_output_list <- list(
  beta_zero = first_half_beta_samples$beta_zero,
  beta_one = first_half_beta_samples$beta_one,
  beta_two = first_half_beta_samples$beta_midpoint[, , 1],
  beta_three = first_half_beta_samples$beta_three,
  beta_five = first_half_beta_samples$beta_midpoint[, , 2],
  beta_six = first_half_beta_samples$beta_six
)

second_half_beta_samples <- extract(model_fit_two, par_names)
second_half_output_list <- list(
  beta_zero = second_half_beta_samples$beta_zero,
  beta_one = second_half_beta_samples$beta_one,
  beta_two = second_half_beta_samples$beta_midpoint[, , 1],
  beta_three = second_half_beta_samples$beta_three,
  beta_five = second_half_beta_samples$beta_midpoint[, , 2],
  beta_six = second_half_beta_samples$beta_six
)

#And then we combine them together into a list:

output_list <- list()

for (beta_name in names(first_half_output_list)) {
  output_list[[beta_name]] <- cbind(
    first_half_output_list[[beta_name]],
    second_half_output_list[[beta_name]]
  )
}

#Then we have a little bit of a hack, to remove some of the samples for
#which the initial amplitude model doesn’t have data points.

index_vec <- as.numeric(rownames(a0_data_no_zeros))
for (ii in 1:length(output_list)) {
  output_list[[ii]] <- output_list[[ii]][, index_vec]
}

#Finally we convert this from a list into an array, so that it can be
#written to h5 (getting the right order of the arguments to dim requires
#thinking about the unlist call, and how array index “running speed”
#works):

output_array <- array(
  as.numeric(unlist(output_list)),
  dim = c(nrow(output_list$beta_zero),
          ncol(output_list$beta_zero),
          length(output_list))
)

# can check that things are as expected:
all(output_array[, , 1] == output_list$beta_zero)

#Lastly we can write them to
#disk:

#output_file <- paste0("./", Sys.Date(), "_beta-samples-array-all-data.h5")
file.h5 <- H5File$new(output_file, mode = "w")
file.h5$create_group("data")
file.h5[["data/beta_samples"]] <- output_array
file.h5$close_all()


