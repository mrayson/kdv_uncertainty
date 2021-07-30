# -*- coding: utf-8 -*-
"""
Create a one cell quad grid

Created on Mon Nov 18 10:23:26 2013

@author: mrayson
"""

import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import operator

from sfoda.ugrid.ugridgen import cartesian_ugrid_gen
from sfoda.suntans.sunboundary import modifyBCmarker, Boundary, InitialCond
from sfoda.suntans.sunpy import Grid

import pdb

PI = np.pi
GRAV=9.81

def make_suntans_kdv(suntanspath, mykdv, bcfunc):
    ####################################################
    # Inputs
    
    # topo paramters
    H = 1000.0
    #h0 = 750.
    #ls = 12500.
    #x0 = 60000.

    # Density parameters
    betas = [1023.7, 1.12, 105, 52, 155, 43] # ~April 5
    #betas = [1023.5, 1.22, 67, 55, 157, 52] # ~March 1

    
    # Boundary forcing parameters
    wave_period=12*3600.

    phi0 = 110.      # Internal wave amplitude (flux m^2/s)
    U0 = 0.0        # Steady flow rate (flux)
    dudz = 0e-4      # Vertical Shear

    # Size of domain
    ny = 1
    nx = 280 # 1km
    nz = 50
    rk = 1.034
    L = 280000 # m

    #suntanspath = 'data'

    starttime = '20000101.000000'
    endtime = '20000130.000000'
    dt = 1800.

    icfile = 'IWave_IC.nc'
    bcfile = 'IWave_BC.nc'
    ####################################################
    def gaussian(x,x0,a,b,pow=2.):
        return b*np.exp(-(x-x0)**pow/(2*a**pow))

    def tanh_hill(x,x0, h0, H, ls):
        return H - (0.5*h0*(1+np.tanh( (x-x0) / (0.5*ls))))

    
    def double_tanh(beta, z):
        return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
            + np.tanh((z+beta[4])/beta[5]))

    #######
    dx = L/nx
    W = dx

    #####
    # Create the grid
    if not os.path.isdir(suntanspath):
        print ('Creating new directory: %s'%suntanspath)
        os.mkdir(suntanspath)
        copyfile('rundata/suntans.dat','%s/suntans.dat'%suntanspath)

    xlims = [0,L]
    ylims = [0,W]

    # Create the grid
    grd= cartesian_ugrid_gen(xlims, ylims, dx, suntanspath=suntanspath)

    # Load the grid
    grd = Grid(suntanspath)

    grd.dv = H*np.ones_like(grd.xv)

    grd.saveBathy('%s/depth.dat-voro'%suntanspath)

    grd.dz = grd.calcVertSpace(nz, rk, H)
    grd.saveVertspace('%s/vertspace.dat'%suntanspath)
    grd.setDepth(grd.dz)

    # Salinity field
    salt = double_tanh(betas, -grd.z_r) - 1000.

    # Create the boundary conditions


    ##########
    # Modify the boundary markers and create the boundary condition file
    ##########

    # Modify the left and right edges and convert to type 2
    grd.mark[grd.mark>0]=1 # reset all edges to type-1

    ## convert edges +/- half a grid cell from the edge
    #dx = hgrd.dg.max()
    xmin = grd.xv.min()-dx/4.0
    xmax = grd.xv.max()+dx/4.0

    grd.calc_edgecoord()

    indleft = operator.and_(grd.mark==1, grd.xe < xmin) # all boundaries
    indright = operator.and_(grd.mark==1, grd.xe > xmax) # all boundaries

    ## Free-surface boundaries
    ##grd.mark[indleft]=3
    ##grd.mark[indright]=3
    #
    ## River boundaries
    grd.mark[indleft]=2
    #grd.mark[indright]=2

    edgefile = suntanspath+'/edges.dat'
    grd.saveEdges(edgefile)


    #Load the boundary object from the grid
    #   Note that this zeros all of the boundary arrays
    bnd = Boundary(suntanspath,(starttime,endtime,dt))

    bnd.setDepth(grd.dv)
    
    t = bnd.tsec-bnd.tsec[0]
    #
    # Velocity boundary
    omega = np.pi*2/wave_period
    for k in range(bnd.Nk):
       for ii, eptr in enumerate(bnd.edgep.tolist()):
           amp = phi0/bnd.de[ii]
           if indleft[eptr]:
               #bnd.boundary_u[:,k,ii] = phi0*np.cos(k_z*grd.z_r[k])*np.sin(omega*t)
               bnd.boundary_u[:,k,ii] = amp*np.sin(omega*t)# + usteady
               bnd.boundary_S[:,k,ii] = salt[k]# - phi0*np.sin(k_z*grd.z_r[k])*np.sin(omega*t)
           elif indright[eptr]:
               bnd.boundary_u[:,k,ii] = amp*np.sin(omega*t)# + usteady
               bnd.boundary_S[:,k,ii] = salt[k]
               #bnd.boundary_u[:,k,ii] = u*nx[ii]
               #bnd.boundary_v[:,k,ii] = u*ny[ii]
               #bnd.boundary_h[:,ii] = h

    ##
    # Set the boundary temperature=1 after a few time steps
    #bnd.boundary_T[4:,:,:] = 1.
    #print('Setting boundary_T=1 at time: ',bnd.time[4])

    # Write the boundary file
    bnd.write2NC(suntanspath+'/'+bcfile)

    #########
    # Create the initial conditions file
    #########
    IC = InitialCond(suntanspath,starttime)

    IC.h[:] = 0

    #IC.T[:,0:nz//2, :] = 1.0 # set T = 1 in the upper water column
    IC.S[:,:,:] = salt[np.newaxis,:,np.newaxis]

    # Write the initial condition file
    IC.writeNC(suntanspath+'/'+icfile,dv=grd.dv)


if __name__=='__main__':
    import sys
    sunpath = sys.argv[1]

    make_suntans(sunpath)

