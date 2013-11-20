''' Library of finite difference matrices
'''
import numpy as np
import scipy as sp
from scipy import sparse

def lap1d(Nx, hx):
    ''' Standard central difference 1d Laplacian with periodic boundary
    conditions
    '''
    diags = np.array([-1,0,1]);    
    dx = np.ones([3, Nx]); dx[1,:] = -2.0*dx[1,:]    
    
    Lx = sp.sparse.spdiags(dx, diags, Nx, Nx, 'lil') 

    # impose periodic boundary conditions
    Lx[0, Nx-1] = 1
    Lx[Nx-1, 0] = 1 
        
    # divide by grid spacings
    Lx = Lx/(hx*hx)
    
    # convert to compressed storage format
    Lx = Lx.tocsr()
   
    return Lx

def lap2d(Nx, Ny, hx, hy):
    ''' Standard 5-point 2d Laplacian with periodic boundary
    conditions
    '''
    diags = np.array([-1,0,1]);    
    dx = np.ones([3, Nx]); dx[1,:] = -2.0*dx[1,:]    
    dy = np.ones([3, Ny]); dy[1,:] = -2.0*dy[1,:]
    
    Lx = sp.sparse.spdiags(dx, diags, Nx, Nx, 'lil') 
    Ly = sp.sparse.spdiags(dy, diags, Ny, Ny, 'lil') 

    # impose periodic boundary conditions
    Lx[0, Nx-1] = 1
    Lx[Nx-1, 0] = 1
    
    Ly[0, Ny-1] = 1
    Ly[Ny-1, 0] = 1
    
    # divide by grid spacings
    Lx = Lx/(hx*hx)
    Ly = Ly/(hy*hy)    
       
    # convert to compressed storage format
    Lx = Lx.tocsr()
    Ly = Ly.tocsr()
        
    # 2d Laplacian operator from Kronecker products of 1d operators
    Ix, Iy = sp.sparse.identity(Nx), sp.sparse.identity(Ny)
    LL = sp.sparse.kron(Iy, Lx) + sp.sparse.kron(Ly, Ix)

    return LL

def deriv2d(Nx, Ny, Nz, hx, hy, hz):
    ''' Standard central-difference 1d derivative with periodic boundary
    conditions, then (delx)^2, (dely)^2, (delz)^2
    '''
    diags = np.array([-1,1]);    
    dx = np.ones([2, Nx]); dx[0,:] = -dx[0,:]    
    dy = np.ones([2, Ny]); dy[0,:] = -dy[0,:]
    dz = np.ones([2, Nz]); dy[0,:] = -dz[0,:]
    
    Lx = sp.sparse.spdiags(dx, diags, Nx, Nx, 'lil') 
    Ly = sp.sparse.spdiags(dy, diags, Ny, Ny, 'lil') 
    Lz = sp.sparse.spdiags(dz, diags, Nz, Nz, 'lil') 
    
    # impose periodic boundary conditions
    Lx[0, Nx-1] = -1
    Lx[Nx-1, 0] = 1
    
    Ly[0, Ny-1] = -1
    Ly[Ny-1, 0] = 1
    
    Lz[0, Nz-1] = -1
    Lz[Nz-1, 0] = 1

    # divide by grid spacings
    Lx = Lx/hx
    Ly = Ly/hy
    Lz = Lz/hz
       
    # convert to compressed storage format
    Lx = Lx.tocsr()
    Ly = Ly.tocsr()
    Lz = Lz.tocsr()
    
    # delx^2, dely^2, delz^2 operator 
    Lxx = sp.sparse.kron(Lx, Lx)
    Lyy = sp.sparse.kron(Ly, Ly)
    Lzz = sp.sparse.kron(Lz, Lz)

    # delxdely, delxdelz, delydelz 
    Lxy = sp.sparse.kron(Ly, Lx) 
    Lxz = sp.sparse.kron(Lz, Lx) 
    Lyz = sp.sparse.kron(Lz, Ly)

    return (Lxx, Lyy, Lzz, Lxy, Lxz, Lyz)


def deriv3d(Nx, Ny, Nz, hx, hy, hz):
    ''' Standard central-difference 1d derivative with periodic boundary
    conditions, then (delx)^2, (dely)^2, (delz)^2
    '''
    diags = np.array([-1,1]);    
    dx = np.ones([2, Nx]); dx[0,:] = -dx[0,:]    
    dy = np.ones([2, Ny]); dy[0,:] = -dy[0,:]
    dz = np.ones([2, Nz]); dz[0,:] = -dz[0,:]

    Lx = sp.sparse.spdiags(dx, diags, Nx, Nx, 'lil') 
    Ly = sp.sparse.spdiags(dy, diags, Ny, Ny, 'lil') 
    Lz = sp.sparse.spdiags(dz, diags, Nz, Nz, 'lil') 
    
    # impose periodic boundary conditions
    Lx[0, Nx-1] = -1
    Lx[Nx-1, 0] = 1
    
    Ly[0, Ny-1] = -1
    Ly[Ny-1, 0] = 1

    Lz[0, Nz-1] = -1
    Lz[Nz-1, 0] = 1

    # divide by grid spacings
    Lx = Lx/hx
    Ly = Ly/hy
    Lz = Lz/hz
       
    # convert to compressed storage format
    Lx = Lx.tocsr()
    Ly = Ly.tocsr()
    Lz = Lz.tocsr()
    
    Ix = sp.sparse.identity(Nx)
    Iy = sp.sparse.identity(Ny)
    Iz = sp.sparse.identity(Nz)

    # delx^2, dely^2, delz^2 operator 
    Lxx = sp.sparse.kron(Ix, sp.sparse.kron(Lx, Lx))
    Lyy = sp.sparse.kron(Iy, sp.sparse.kron(Ly, Ly))
    Lzz = sp.sparse.kron(Iz, sp.sparse.kron(Lz, Lz))

    # delxdely, delxdelz, delydelz 
    Lxy = sp.sparse.kron(Iz, sp.sparse.kron(Ly, Lx)) 
    Lxz = sp.sparse.kron(Iy, sp.sparse.kron(Lz, Lx)) 
    Lyz = sp.sparse.kron(Ix, sp.sparse.kron(Lz, Ly))

    return (Lxx, Lyy, Lzz, Lxy, Lxz, Lyz)
            
def lap3d(Nx, Ny, Nz, hx, hy, hz):
    ''' Standard 7-point 3d Laplacian with periodic boundary
    conditions
    '''    
    diags = np.array([-1,0,1]);    
    dx = np.ones([3, Nx]); dx[1,:] = -2.0*dx[1,:]    
    dy = np.ones([3, Ny]); dy[1,:] = -2.0*dy[1,:]
    dz = np.ones([3, Nz]); dz[1,:] = -2.0*dz[1,:]

    Lx = sp.sparse.spdiags(dx, diags, Nx, Nx, 'lil') 
    Ly = sp.sparse.spdiags(dy, diags, Ny, Ny, 'lil') 
    Lz = sp.sparse.spdiags(dz, diags, Nz, Nz, 'lil') 

    # impose periodic boundary conditions
    Lx[0, Nx-1] = 1
    Lx[Nx-1, 0] = 1
    
    Ly[0, Ny-1] = 1
    Ly[Ny-1, 0] = 1
    
    Lz[0, Nz-1] = 1
    Lz[Nz-1, 0] = 1
   
    # divide by grid spacings
    Lx = Lx/(hx*hx)
    Ly = Ly/(hy*hy)
    Lz = Lz/(hz*hz)   
   
    # convert to compressed storage format
    Lx = Lx.tocsr()
    Ly = Ly.tocsr()
    Lz = Lz.tocsr()

    Ix = sp.sparse.identity(Nx)
    Iy = sp.sparse.identity(Ny)
    Iz = sp.sparse.identity(Nz)

    # 3d Laplacian operator from Kronecker products of 1d operators    
    LL = sp.sparse.kron(Iz, sp.sparse.kron(Iy, Lx)) \
       + sp.sparse.kron(Iz, sp.sparse.kron(Ly, Ix)) \
       + sp.sparse.kron(sp.sparse.kron(Lz,Iy),Ix)
       
    return LL