dat_1 <- read.csv("/home/suntans/Share/ARCHub/DATA/FIELD/IMOS/IMOS_Density_KIM200_2013_a.csv", header = F)
col_names_1 <- dat_1[1,]
dat_1 <- dat_1[-c(1:11, 535:(nrow(dat_1))),] # hard coded for this csv file
colnames(dat_1) <- c("Time", as.character(col_names_1)[-1])
rownames(dat_1) <- 1:nrow(dat_1)
head(dat_1, 20)

#getwd()
#setwd("./ddcurves2/")

# depth limits
x_min <- 0
x_max <- -205.0

depths_1 <- as.numeric(col_names_1[,-1])

IMOS_KIM200_2013a <- list(min_depth = 0,
                        max_depth = -205.0,
                        depths = depths_1, 
                        Time = dat_1[,1], 
                        densitiy_mat = dat_1[,-1], 
                        full_dat = dat_1)

save(IMOS_KIM200_2013a, file = "./IMOS_KIM200_2013a.Rdata")
load("./IMOS_KIM200_2013a.Rdata")

###
dat_1 <- read.csv("/home/suntans/Share/ARCHub/DATA/FIELD/IMOS/IMOS_Density_ITFTIS_2015_b.csv", header = F)
col_names_1 <- dat_1[1,]
dat_1 <- dat_1[-c(1:11, 741:(nrow(dat_1))),] # hard coded for this csv file
colnames(dat_1) <- c("Time", as.character(col_names_1)[-1])
rownames(dat_1) <- 1:nrow(dat_1)
head(dat_1, 20)

#getwd()
#setwd("./ddcurves2/")

# depth limits
x_min <- 0
x_max <- -470.0

depths_1 <- as.numeric(col_names_1[,-1])

IMOS_ITFTIS_2015_b <- list(min_depth = 0,
                          max_depth = -470.0,
                          depths = depths_1, 
                          Time = dat_1[,1], 
                          densitiy_mat = dat_1[,-1], 
                          full_dat = dat_1)

save(IMOS_KIM200_2013a, file = "./IMOS_ITFTIS_2015_b.Rdata")
load("./IMOS_ITFTIS_2015_b.Rdata")

