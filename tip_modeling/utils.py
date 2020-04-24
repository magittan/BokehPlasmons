import os,h5py
import numpy as np
from scipy import special as sp
from common.baseclasses import ArrayWithAxes as AWA

def progress(i,L,last):
    next = last+10
    percent = 100*i/L
    if percent >= next:
        print('\t{}% complete...'.format(next))
        return next
    else:
        return last

def load_eigpairs(basedir=os.path.dirname("./"),eigpair_fname="UnitSquareMesh_100x100_1000_eigenbasis.h5"):
    """Normalization by sum always ensures that integration will be like summing, which is
    much simpler than keeping track of dx, dy..."""

    global eigpairs
    eigpairs = dict()

    path=os.path.join(basedir,eigpair_fname)

    with h5py.File(path,'r') as f:
        for key in list(f.keys()):
            eigfunc=np.array(f.get(key))
            eigfunc/=np.sqrt(np.sum(np.abs(eigfunc)**2))
            eigpairs[float(key)] = AWA(eigfunc,\
                                       axes=[np.linspace(0,1,eigfunc.shape[0]),\
                                             np.linspace(0,1,eigfunc.shape[1])])
    return eigpairs

def inner_prod(psi,psi_star):
    return np.sum(psi*psi_star)

def dipole_field(x,y,z,direction=[0,1]):
    "`direction` is a vector with `[\rho,z]` components"

    r=np.sqrt(x**2+y**2+z**2)
    rho=np.sqrt(x**2+y**2)
    rhat_rho=rho/r
    rhat_z=z/r

    return (direction[0]*rhat_rho+direction[1]*rhat_z)/r**2

def unscreened_coulomb_kernel_fourier(kx,ky):
    return np.where((kx==0)*(ky==0),0,2*np.pi/np.sqrt(kx**2+ky**2))

def bessel(A,v,Q,x,y):
    r = np.sqrt(x**2+y**2)
    return A*sp.jv(v,Q*r)

def planewave(qx,qy,x,y,x0=0,y0=0,phi0=0):
    return np.sin(qx*(x-x0)+qy*(y-y0)+phi0)

def faketip(q,n,N,x,y):
    Q = q*(n+1)
    exp_prefactor = np.exp(-2*(n+1)/(N+1))
    
    D=1#The larger D goes, the longer range the tip excitation
    func = lambda x,y: np.exp(-q**2/(2*D**2)*(x**2+y**2))*bessel(1,0,Q,x,y)

    return func
