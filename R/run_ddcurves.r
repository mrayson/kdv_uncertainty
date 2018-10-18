#!/usr/bin/Rscript
library(ddcurves2)
library(rstan)
library(hdf5r)

load_csv <- function(csvfile, z_max){
    #Load density data from a csv file
    #
    dat_1 <- read.csv(csvfile, header = F)
    dat_1 <- dat_1[,-nrow(dat_1)]
    col_names_1 <- dat_1[1,]
    dat_1 <- dat_1[-1,]
    colnames(dat_1) <- c("Time", as.character(col_names_1)[-1])
    rownames(dat_1) <- 1:nrow(dat_1)
    head(dat_1)


    depths_1 <- as.numeric(col_names_1[,-1])

    csv_data <- list(min_depth = 0,
                    max_depth = z_max,
                    depths = depths_1, 
                    Time = dat_1[,1], 
                    densitiy_mat = dat_1[,-1], 
                    full_dat = dat_1)

    return(csv_data)

}

run_ddcurves <- function(infile, max_depth){
    ###
    #infile <- "DATA_QC/IMOS_Density_KIM200_2013_a"
    #infile <- "DATA_QC/IMOS_test"
    #max_depth <- -205.0
    #infile <- "DATA_QC/Crux_KP150_Phs2_Density_lowpass"
    #infile <- "DATA_QC/Crux_test"
    #max_depth <- -252.5
    ###

    csv_file = paste0(infile, ".csv")

    csv_data = load_csv(csv_file, max_depth)

    print(csv_data$densitiy_mat)

    stan_dat_one <- list(
      n_depths = length(csv_data$depths),
      n_times = nrow(csv_data$densitiy_mat),
      depths = csv_data$depths,
      densities = csv_data$densitiy_mat
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
      #beta_midpoint = array(c(rep(30, n_t_one), rep(150, n_t_one)), dim = c(n_t_one, 2)),
      mean_beta_zero = 1025,
      mean_beta_one = 1,
      mean_beta_three = 42,
      mean_beta_six = 48,
      mean_beta_midpoint = array(c(77, 150), dim = 2),
      sigma_beta = array(c(0.18, 0.12, 26.1, 8.09, 12.6, 6.65), dim = 6),
      #sigma_beta = array(c(0.18, 0.12, 26.1, 8.09, 24.6, 6.65), dim = 6),
      
      sigma_curve = 0.08
    )

    #The density profile models are then fit:

    model_fit_one <- double_tanh_no_timeseries(
      stan_data_list = stan_dat_one,
      iter = 2500, # add an extra zero to this later, and the following
      warmup = 2000,
      save_warmup = FALSE,
      chains = 3,
      cores = 3,
      refresh = 25,
      control = list(adapt_delta = 0.95, max_treedepth = 16, stepsize = 1e-6),
      init = list(init_sub, init_sub, init_sub)
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

    ##And then we combine them together into a list:
    #
    #output_list <- list()
    #
    #for (beta_name in names(first_half_output_list)) {
    #  output_list[[beta_name]] <- cbind(
    #    first_half_output_list[[beta_name]],
    #    second_half_output_list[[beta_name]]
    #  )
    #}
    output_list <- first_half_output_list

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

    output_file <- paste0(infile, "_beta-samples-array-all-data.h5")
    file.h5 <- H5File$new(output_file, mode = "w")
    file.h5$create_group("data")
    file.h5[["data/beta_samples"]] <- output_array
    file.h5[["data/time"]] <- csv_data$Time
    file.h5$close_all()

}

##############################
# Command Line Arguments
##############################
args = commandArgs(trailingOnly = TRUE)

if (length(args)!=2) {
  stop("Usage example: run_ddcurves.r infile_name -200.n", call.=FALSE)
} else  {
    # default output file
      infile <- args[1]
      depth <- args[2]
      run_ddcurves(infile,depth)
}

