"""
Test the KdV-->SUNTANS conversion functions
"""

import os
import numpy as np
import xarray as xr
from kdvutils import load_beta_h5, init_vkdv_a0, bcfunc

from sfoda.ugrid.ugridgen import cartesian_ugrid_gen
from sfoda.suntans.sunboundary import Boundary, InitialCond
from sfoda.suntans.sunpy import Grid

from iwaves.utils import density
from iwaves.utils.tools import grad_z
from iwaves.utils.isw import iwave_modes_uneven


### 
# Input variables
def make_suntans_kdv(suntanspath, t1, t2, draw_num):
    print(suntanspath, t1, t2, draw_num)
    print(t1.tolist().strftime('%Y%m%d'))
    print(t2.tolist().strftime('%Y%m%d'))

    beta_infile = 'inputs/ShellCrux_Filtered_Density_Harmonic_MCMC_20162017_prediction.h5'
    a0_infile = 'inputs/a0_samples_harmonicfit_M2S2N2K1O1_na3_dt60min_12month.nc'
    kdvfile = 'data/kdvin.yml'
    depthfile = 'data/kdv_bathy_Prelude_WELGA_bathy_1km.csv'
    mode = 0

    # vertical grid
    nz = 50
    rk = 1.04
    dt = 1200
    icfile = 'IWave_IC.nc'
    bcfile = 'IWave_BC.nc'

    ###########################################

    #suntanspath = suntanspath_base.format(draw_num)
    #print(suntanspath)

    # Load the beta and a0 files
    beta_ds = load_beta_h5(beta_infile)
    a0_ds = xr.open_dataset(a0_infile, group='predictions')

    mykdv, F_a0, t0, runtime, density_params, twave, ampfac = \
        init_vkdv_a0(depthfile, kdvfile, beta_ds, a0_ds, draw_num, t1, t2)

    ######
    ## Create the grid
    if not os.path.isdir(suntanspath):
        print ('Creating new directory: %s'%suntanspath)
        os.mkdir(suntanspath)
        #copyfile('rundata/suntans.dat','%s/suntans.dat'%suntanspath)


    dx = mykdv.dx
    xlims = [mykdv.x[0], mykdv.x[-1]]
    ylims = [0, dx]

    # Create the grid
    grd= cartesian_ugrid_gen(xlims, ylims, mykdv.dx, suntanspath=suntanspath)

    # Load the grid
    grd = Grid(suntanspath)

    grd.dv = 0.5*(mykdv.h[1:] + mykdv.h[:-1])

    grd.saveBathy('%s/depth.dat-voro'%suntanspath)

    grd.dz = grd.calcVertSpace(nz, rk, mykdv.h.max())
    grd.saveVertspace('%s/vertspace.dat'%suntanspath)
    grd.setDepth(grd.dz)

    # Salinity field
     
    # Comput the modes

    def mid_extrap(A, nk):
        B = np.zeros((nk+1,), dtype=A.dtype)
        B[1:-1] = 0.5*A[0:-1]+0.5*A[1::]
        B[0] = B[1]
        B[-1] = B[-2]
        return B

    nk = nz
    z = grd.z_r[0:nk]
    zw = np.zeros((nk+1,))
    zw[1:] = np.cumsum(grd.dz)

    salt = density.double_tanh_rho_new2(-z, *density_params) - 1000.
    saltw = density.double_tanh_rho_new2(-zw, *density_params) - 1000.

    N2 = -9.81/1024 * grad_z(saltw, -zw)

    # Calculate the mode shapes
    phi, cn = iwave_modes_uneven(N2, zw)
    phi_z_w = grad_z(phi[:,mode], -zw)
    phi_z = 0.5*(phi_z_w[1:] + phi_z_w[:-1])

    print(mykdv.c[0], cn[mode])

    # Create the boundary object here...
    ##########
    # Modify the boundary markers and create the boundary condition file
    ##########
    starttime = t0.tolist().strftime('%Y%m%d.%H%M')
    endtime = t2.tolist().strftime('%Y%m%d.%H%M')

    # Modify the left and right edges and convert to type 2
    grd.mark[grd.mark>0]=1 # reset all edges to type-1

    ## convert edges +/- half a grid cell from the edge
    #dx = hgrd.dg.max()
    xmin = grd.xv.min()-dx/4.0
    xmax = grd.xv.max()+dx/4.0

    grd.calc_edgecoord()

    indleft = (grd.mark==1) & (grd.xe < xmin) # all boundaries
    indright = (grd.mark==1) & (grd.xe > xmax) # all boundaries

    ## Free-surface boundaries
    ##grd.mark[indleft]=3
    ##grd.mark[indright]=3
    #
    ## River boundaries
    grd.mark[indleft]=2
    grd.mark[indright]=2

    edgefile = suntanspath+'/edges.dat'
    grd.saveEdges(edgefile)


    #Load the boundary object from the grid
    #   Note that this zeros all of the boundary arrays
    bnd = Boundary(suntanspath,(starttime,endtime,dt))

    bnd.setDepth(grd.dv)

    t = bnd.tsec-bnd.tsec[0]
    #
    # Velocity boundary
    amp = bcfunc(F_a0, t, 6*3600, twave=twave)
    c1 = cn[mode]
    for k in range(bnd.Nk):
       for ii, eptr in enumerate(bnd.edgep.tolist()):
           if indleft[eptr]:
               bnd.boundary_u[:,k,ii] = amp*c1*phi_z[k]
               bnd.boundary_S[:,k,ii] = salt[k]
           elif indright[eptr]:
               bnd.boundary_S[:,k,ii] = salt[k]

    ##

    # Write the boundary file
    bnd.write2NC(suntanspath+'/'+bcfile)

    #########
    # Create the initial conditions file
    #########
    IC = InitialCond(suntanspath,starttime)

    IC.S[:,:,:] = salt[np.newaxis,:,np.newaxis]

    # Write the initial condition file
    IC.writeNC(suntanspath+'/'+icfile,dv=grd.dv)

if __name__=='__main__':
    import sys
    sunpath = sys.argv[1]
    t1 = np.datetime64(str(sys.argv[2]))
    t2 = np.datetime64(str(sys.argv[3]))

    draw_num = int(sys.argv[4])


    make_suntans_kdv(sunpath, t1, t2, draw_num)


