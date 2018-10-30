FROM ubuntu:17.10

LABEL maintainer="matt.rayson@uwa.edu.au"

#######################################################
#
# Python 3 with mpi docker
#
# Build via:
#  docker build -t mpi-python .
#

# Add aarnet mirror to speed up package update
RUN perl -p -i.orig -e \
      's/archive.ubuntu.com/mirror.aarnet.edu.au\/pub\/ubuntu\/archive/' /etc/apt/sources.list \
      && sed -i '0,/# deb-src/{s/# deb-src/deb-src/}' /etc/apt/sources.list

# Install package dependencies
RUN apt-get update \
      && apt-get install -y \
         build-essential \
         gdb \
         gfortran \
         python3.6 \
         python3-pip \
         wget \
      && apt-get clean all \
      && rm -r /var/lib/apt/lists/*

#RUN alias pip3="python3 -m pip"

# pip numpy scipy etc
# Note that pandas/xarray get h5py and netCDF4 libs
RUN pip3 install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    xarray\
    gsw


### Build MPICH ###

ARG MPICH_VERSION="3.1.4"
ARG MPICH_CONFIGURE_OPTIONS="--enable-fast=all,O3 --prefix=/usr"
ARG MPICH_MAKE_OPTIONS="-j4"

WORKDIR /tmp/mpich-build

RUN wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz \
      && tar xvzf mpich-${MPICH_VERSION}.tar.gz \
      && cd mpich-${MPICH_VERSION}  \
      && ./configure ${MPICH_CONFIGURE_OPTIONS} \
      && make ${MPICH_MAKE_OPTIONS} \
      && make install \
      && ldconfig

## Test MPICH
#WORKDIR /tmp/mpich-test
#COPY mpich-test .
#RUN sh test.sh


### Build MPI4PY ###

ARG MPI4PY_VERSION="3.0.0"

WORKDIR /tmp/mpi4py-build

RUN wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.0.tar.gz \
      && tar xzvf mpi4py-${MPI4PY_VERSION}.tar.gz \
      && cd mpi4py-${MPI4PY_VERSION} \
      && python3.6 setup.py build \
      && python3.6 setup.py install \
      && rm -r /tmp/mpi4py-build


### Build OSU Benchmarks ###

ARG OSU_BENCH_VERSION="5.4.2"
ARG OSU_BENCH_CONFIGURE_OPTIONS="--prefix=/usr/local CC=mpicc CXX=mpicxx CFLAGS=-O3"
ARG OSU_BENCH_MAKE_OPTIONS="-j4"

WORKDIR /tmp/osu-benchmark-build

RUN wget http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_BENCH_VERSION}.tar.gz \
      && tar xzvf osu-micro-benchmarks-${OSU_BENCH_VERSION}.tar.gz \
      && cd osu-micro-benchmarks-${OSU_BENCH_VERSION} \
      && ./configure ${OSU_BENCH_CONFIGURE_OPTIONS} \
      && make ${OSU_BENCH_MAKE_OPTIONS} \
      && make install

WORKDIR /
RUN rm -rf /tmp/*


# copy this package into the image and create directories for input and output
#ADD . SOLITON
#ENV SOLITON_HOME /SOLITON
#RUN chmod u+x SOLITON/python_utils/run_pde_solver.py
#RUN mkdir SOLITON/output
#RUN mkdir SOLITON/output/slim
#RUN mkdir SOLITON/output/full
#
#RUN mkdir SOLITON/inputs

## iwaves
#RUN wget https://bitbucket.org/mrayson/iwaves/get/v0.2.0a.tar.gz \
#     && tar xvzf v0.2.0a.tar.gz -C SOLITON \
#     && mv SOLITON/mrayson-iwaves-6ea57f5a55f5 SOLITON/iwaves

RUN wget https://bitbucket.org/mrayson/iwaves/get/v0.2.0a.tar.gz \
     && tar xvzf v0.2.0a.tar.gz  \
     && mv mrayson-iwaves-6ea57f5a55f5 iwaves

# input a0 and beta files need to be in this directory (or a subdir)
# in order to copy them to docker image

#COPY <input_a0_file> SOLITON/inputs
#COPY <input_beta_file> SOLITON/inputs

## azure config
#ENV AZURE_ACCOUNT <azure_account_name>
#ENV AZURE_KEY <azure_account_key>
#ENV AZURE_CONTAINER <azure_storage_container>

#RUN python3.6 -c "print('hello')"
#RUN chmod u+x SOLITON/test.py

#WORKDIR SOLITON

CMD ["/bin/bash"]

