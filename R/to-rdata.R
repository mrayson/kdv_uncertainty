dat_1 <- read.csv("./../../../soli_modelling/data/2018-03-09_Crux_yearlong_initial/Crux_KP150_Phs1_Density_lowpass.csv", header = F)
dat_1 <- dat_1[,-13]
col_names_1 <- dat_1[1,]
dat_1 <- dat_1[-1,]
colnames(dat_1) <- c("Time", as.character(col_names_1)[-1])
rownames(dat_1) <- 1:nrow(dat_1)
head(dat_1)

getwd()
setwd("./dpcurves/")
# depth limits
x_min <- 0
x_max <- -252.5

depths_1 <- as.numeric(col_names_1[,-1])

Crux_KP150_Phs1 <- list(min_depth = 0,
                        max_depth = -252.5,
                        depths = depths_1, 
                        Time = dat_1[,1], 
                        pressure_mat = dat_1[,-1], 
                        full_dat = dat_1)

# save(Crux_KP150_Phs1, file = "./data/Crux_KP150_Phs1.Rdata")
save(Crux_KP150_Phs1, file = "../../data/2018-03-09_Crux_yearlong_initial/Crux_KP150_Phs1.Rdata")

dat_2 <- read.csv("./../../../soli_modelling/data/2018-03-09_Crux_yearlong_initial/Crux_KP150_Phs2_Density_lowpass.csv", header = F)
dat_2 <- dat_2[,-14]
col_names_2 <- dat_2[1,]
dat_2 <- dat_2[-1,]
colnames(dat_2) <- c("Time", as.character(col_names_2)[-1])
rownames(dat_2) <- 1:nrow(dat_2)
head(dat_2)

depths_2 <- as.numeric(col_names_2[,-1])

Crux_KP150_Phs2 <- list(min_depth = 0,
                        max_depth = -252.5,
                        depths = depths_2, 
                        Time = dat_2[,1], 
                        pressure_mat = dat_2[,-1], 
                        full_dat = dat_2)

# save(Crux_KP150_Phs2, file = "./data/Crux_KP150_Phs2.Rdata")
save(Crux_KP150_Phs2, file = "../../data/2018-03-09_Crux_yearlong_initial/Crux_KP150_Phs2.Rdata")

