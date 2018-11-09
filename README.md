# Scripts and notebooks for nonlinear internal wave uncertainty analysis study

Used to generate figures and test ideas for the following papers:

Manderson et al., 2018, Uncertainty quantification of density and stratification estimates with implications for predicting ocean dynamics, **to be submitted to JAOTEC**

Rayson et al., 2018, Prediction of nonlinear internal wave amplitude in shelf seas with uncertainty quantification, **in preparation**

## Other repos

The following repos are linked to this project:

- [https://bitbucket.org/mrayson/iwaves] KdV solver code
- [https://github.com/hhau/ddcurves2] Bayesian inference code (R, Stan) for density-depth profiles (private)
- [https://github.com/alan-turing-institute/Soliton] Code for running kdv solver in the azure cloud and shiny dashboard for viewing results. (private)

## Docker run help

 - Build the docker file locally and push the container to dockerhub

    `sudo docker build -t iwaves .`

 - Test it runs

    `sudo docker run iwaves python run_pde_solver_mpi.py`
    (It will crash due to not finding a file but should load the libraries fine...)

 - Push to docker hub   
    
    `sudo docker tag iwaves latest`
    `sudo docker push mrayson/iwaves`

---

Matt Rayson

University of Western Australia

October 2018



