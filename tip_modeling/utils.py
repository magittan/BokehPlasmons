import os,h5py
import time
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
    
class Timer(object):
    
    def __init__(self):
        
        self.t0=time.time()
        
    def __call__(self,reset=True):
        
        t=time.time()
        print('\tTime elapsed:',t-self.t0)
        
        if reset: self.t0=t

def inner_prod(psi1,psi2):
    
    psi1=np.asarray(psi1)
    psi2=np.asarray(psi2)
    
    return np.sum(np.conj(psi1)*psi2)

def normalize(psi):
    
    return psi/np.sqrt(inner_prod(psi,psi))

def align_to_reals(psi0):
    """This algorithm applies an overall phase to align
    a complex vector with the real axis, in an average sense."""
    
    psi0=np.asarray(psi0)
    R2=np.sum(psi0**2)/np.sum(np.conj(psi0)**2)
    
    # Phase is defined only up to pi/2
    p=1/4*np.angle(R2); psi=psi0*np.exp(-1j*p)
    
    #Check if we chose correct phase
    Nr=np.real(np.sum(np.abs(psi+np.conj(psi))**2)/4)
    N=np.real(inner_prod(psi,psi))
    Ni=np.real(np.sqrt(N**2-Nr**2))
    realness=Nr/Ni
    
    if Ni and realness<1: psi*=np.exp(-1j*np.pi/2)
    
    return psi

# This kind of makes `inner_prod` redundant
def build_matrix(functions1,functions2):
    """
    Builds matrix of inner products between two lists of functions.
    TODO: this function is where all the meat is, so probably need to optimize.

    Parameters
    ----------
    functions1 : list of 2D `np.ndarray`
        Represents a basis of functions.
    functions2 : list of 2D `np.ndarray`
        Represents a basis of functions.

    Returns
    -------
    M_mn : `np.matrix`
        Element m,n corresponds with inner product
        < `functions1[m]` | `functions2[n]` >.

    """
    
    Nfunctions=len(functions1)
    assert len(functions1)==len(functions2)
    
    print('Building %ix%i matrix...'%(Nfunctions,Nfunctions))

    T=Timer()
    U1=np.array([func.ravel() for func in functions1])
    U2=np.array([func.ravel() for func in functions2])
    
    M_mn = np.matrix(np.conj(U1) @ U2.T)
    T()
    
    return M_mn

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
    
    D=1#The larger D goes, the longer range the tip excitation
    func = lambda x,y: np.exp(-q**2/(2*D**2)*(x**2+y**2))*bessel(1,0,Q,x,y)

    return func
