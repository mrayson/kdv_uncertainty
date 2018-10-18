load_csv <- function(csvfile, z_max){
    dat_1 <- read.csv(csvfile, header = F)
    col_names_1 <- dat_1[1,]
    dat_1 <- dat_1[-c(1:11, 535:(nrow(dat_1))),] # hard coded for this csv file
    colnames(dat_1) <- c("Time", as.character(col_names_1)[-1])
    rownames(dat_1) <- 1:nrow(dat_1)
    head(dat_1, 20)


    depths_1 <- as.numeric(col_names_1[,-1])

    csv_data <- list(min_depth = 0,
                    max_depth = z_max,
                    depths = depths_1, 
                    Time = dat_1[,1], 
                    densitiy_mat = dat_1[,-1], 
                    full_dat = dat_1)

    return(csv_data)

}

csv_data = load_csv("/home/suntans/Share/ARCHub/DATA/FIELD/IMOS/IMOS_Density_KIM200_2013_a.csv", -205.0)
print(csv_data$depths)
print(csv_data$Time)
