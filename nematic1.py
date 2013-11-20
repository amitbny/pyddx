"""
Nematic hydrodynamics in Darcy's approximation
Ingredients:Uniaxial/Biaxial nematic with isotropic/anisotropic derivatives
copyright: amitb@courant.nyu.edu

"""
import numpy as np  # multidimensional arrays, linear algebra,...
import scipy as sp  # signal and image processing library
import scipy.io  
import matplotlib as mpl        # 2D/3D plotting library
import matplotlib.pyplot as plt 
from pylab import *             
import time                     # for system simulation time
import pyddx                    # for finite difference derivatives
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mayavi.mlab import *

clf(); close(); 

#===========================#
# set computation parameters
#===========================#

# grid spacing and size
hx = 1; hy = 1; hz = 1; Nx = 64; Ny = 64; Nz = 64;
# order parameter size
Na, N   = 5, Nx*Ny*Nz        
# parameters of local part of Landau-Ginzburg functional
A = -0.08; B = -0.5;  C = 2.66; Ep = 0 
# parameters of gradient part of Landau-Ginzburg functional
L1 = 1.0; L2 = 0.0            
# mobility parameter, time step, and integration time
Gamma = 0.1; dt = 1.0; tsteps = 1e2; tmax = dt*tsteps;

#========================================================#
# finite difference Laplacian matrix from pyddx in 2D/3D
#========================================================#

#LL = fdlib.lap2d(Nx, Ny, hx, hy)
#Lxx, Lyy, Lzz, Lxy, Lxz, Lyz = fdlib.deriv2d(Nx, Ny, Nz, hx, hy, hz)

LL = pyddx.lap3d(Nx, Ny, Nz, hx, hy, hz)
Lxx, Lyy, Lzz, Lxy, Lxz, Lyz = pyddx.deriv3d(Nx, Ny, Nz, hx, hy, hz)

# initialise the nematic variables to random value and set fx to zero
a  = 0.1*np.random.randn(N, Na); fx = np.zeros(shape=(N, Na));

#=======================#
#start time integration
#=======================#

start = time.clock()

for step in arange(0, tsteps):

    # compute AA=A+CTrQ^2 and BB=B+6EprimeTrQ^3    
    AA  = Gamma*(A + C*np.square(a).sum(axis=1));
    BB  = Gamma*(B + Ep*( sqrt(6.0)*np.square(a[:,0])*a[:,0] + 
          sqrt(27.0/2.0)*a[:,0]*(- 2.0*np.square(a[:,1]) - 
          2.0*np.square(a[:,2]) + np.square(a[:,3]) + np.square(a[:,4])) + 
          9.0/sqrt(2.0)*( 2.0*a[:,2]*a[:,3]*a[:,4] + a[:,1]*( 
          np.square(a[:,3]) - np.square(a[:,4])) )))
          
    # compute isotropic/anisotropic derivatives       
    fa1 = Gamma*L1*LL.dot(a[:,0].T) + Gamma*L2*(LL.dot(a[:,0].T)/6.0 + 
          0.5*Lzz.dot(a[:,0].T) + ((Lyy-Lxx).dot(a[:,1].T) + 
          Lxz.dot(a[:,3].T) + Lyz.dot(a[:,4].T))/sqrt(12.0) - 
          Lxy.dot(a[:,2].T)/sqrt(3.0));
    
    fa2 = Gamma*L1*LL.dot(a[:,1].T) + Gamma*L2*((Lyy-
          Lxx).dot(a[:,0].T)/sqrt(12.0) + 0.5*((Lxx+Lyy).dot(a[:,1].T) + 
          Lxz.dot(a[:,3].T) - Lyz.dot(a[:,4].T)));

    fa3 = Gamma*L1*LL.dot(a[:,2].T) + Gamma*L2*(-Lxy.dot(a[:,0].T)/sqrt(3.0)
          + 0.5*((Lxx+Lyy).dot(a[:,2].T) + Lyz.dot(a[:,3].T) + 
          Lxz.dot(a[:,4].T)));

    fa4 = Gamma*L1*LL.dot(a[:,3].T) + Gamma*L2*(Lxz.dot(a[:,0].T)/sqrt(12.0)
          + 0.5*((Lxx+Lzz).dot(a[:,3].T) + Lxz.dot(a[:,1].T) + 
          Lyz.dot(a[:,2].T) + Lxy.dot(a[:,4].T)));

    fa5 = Gamma*L1*LL.dot(a[:,4].T) + Gamma*L2*(Lyz.dot(a[:,0].T)/sqrt(12.0)
          + 0.5*((Lyy+Lzz).dot(a[:,4].T) - Lyz.dot(a[:,1].T) + 
          Lxz.dot(a[:,2].T) + Lxy.dot(a[:,3].T)));
    
    # assemble local and nonlocal parts together
    fx[:,0] = fa1 - AA*a[:,0] - BB/sqrt(6.0)* \
              (np.square(a[:,0]) - np.square(a[:,1]) - np.square(a[:,2]) 
              + 0.5*(np.square(a[:,3]) + np.square(a[:,4])));

    fx[:,1] = fa2 - AA*a[:,1] - BB*(-sqrt(2.0/3.0)*a[:,0]*a[:,1]  
              + sqrt(1.0/8.0)*(np.square(a[:,3]) - np.square(a[:,4])));

    fx[:,2] = fa3 - AA*a[:,2] - BB*(-2.0/sqrt(6.0)*a[:,0]*a[:,2]  
              + sqrt(1.0/2.0)*a[:,3]*a[:,4]);
    
    fx[:,3] = fa4 - AA*a[:,3] - BB*(sqrt(1.0/6.0)*a[:,0]*a[:,3] 
              + sqrt(1.0/2.0)*(a[:,1]*a[:,3] + a[:,2]*a[:,4]));
    
    fx[:,4] = fa5 - AA*a[:,4] - BB*(sqrt(1.0/6.0)*a[:,0]*a[:,4] 
              + sqrt(1.0/2.0)*(a[:,2]*a[:,3] - a[:,1]*a[:,4]));
    
    # Euler integration
    a = a + dt*fx
    print step
    
    if (step%100 == 0):
        
        # for 2D         
        #ax = pcolor(np.reshape(np.sum(a*a, axis=1), (Nx, Ny)));
        #axis('tight'); 
        #divider = make_axes_locatable(gca())
        #cax = divider.append_axes("right", "5%", pad="3%")
        #colorbar(ax, cax=cax);
        
        # for 3D   
        #ax = pipeline.iso_surface(np.reshape(np.sum(a*a, axis=1), (Nx, Ny, Nz)))
        ax = contour3d(np.reshape(np.sum(a*a, axis=1), (Nx, Ny, Nz)))
        colorbar(ax, orientation='vertical')
        savefig('fig.jpg'); #ax = pcolor(b[:,:,Nz/2]) 
        pause(.01)
    
stop = time.clock()
print stop - start

#mlab.contour3d(np.reshape(np.sum(a*a, axis=1), (Nx, Ny, Nz)))
#mlab.pipeline.iso_surface(np.reshape(np.sum(a*a, axis=1), (Nx, Ny, Nz)))
show()