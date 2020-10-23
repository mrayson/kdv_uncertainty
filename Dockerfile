FROM mrayson/mpi-python:latest

LABEL maintainer="matt.rayson@uwa.edu.au"

#######################################################
#
# Python 3 with mpi docker
#
# Build via:
#  docker build -t mpi-python .
#

# Get the MKL numpy, etc
#RUN pip3 uninstall numpy scipy -y
#RUN pip3 install \
#    intel-numpy \
#    intel-scipy

# copy this package into the image and create directories for input and output
#ADD . SOLITON
#ENV SOLITON_HOME /SOLITON
#RUN chmod u+x SOLITON/python_utils/run_pde_solver.py

# Not necessary on shifter
#ADD scripts /SOLITON
#RUN chmod u+x SOLITON/run_pde_solver_mpi.py

#RUN mkdir SOLITON/output
#RUN mkdir SOLITON/output/slim
#RUN mkdir SOLITON/output/full
#RUN mkdir SOLITON/inputs

## iwaves
#RUN wget https://bitbucket.org/mrayson/iwaves/get/v0.2.0a.tar.gz \
#     && tar xvzf v0.2.0a.tar.gz -C SOLITON \
#     && mv SOLITON/mrayson-iwaves-6ea57f5a55f5 SOLITON/iwaves

# Use pip
RUN pip install pyyaml
RUN conda install -y git 
RUN pip install git+https://bitbucket.org/mrayson/iwaves.git@master
RUN pip install git+https://github.com/mrayson/soda.git@python3pip
#
# input a0 and beta files need to be in this directory (or a subdir)
# in order to copy them to docker image
#COPY inputs/2018-05-22_a0-samples-at-all-times.h5 SOLITON/inputs
#COPY inputs/2018-05-22_beta-samples-array-all-data.h5 SOLITON/inputs

#ENV PYTHONPATH /SOLITON
#
#WORKDIR SOLITON

#RUN python3.6 -c "import iwaves;print('iwaves successfully imported')"
#RUN python -c "import numpy as np;np.__config__.show();import iwaves"
#RUN ls inputs/

#CMD ["/bin/bash"]

