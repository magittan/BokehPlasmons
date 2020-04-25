# -*- coding: utf-8 -*-
import os
import multiprocessing as mp
import numpy as np
import pickle
from itertools import product
from common.baseclasses import ArrayWithAxes as AWA
from BokehPlasmons.tip_modeling.utils import build_matrix,align_to_reals,Timer,normalize


def load_eigpairs(name='UnitSquareMesh_101x101x2000_Neumann_eigenbasis.pickle'):
    
    filepath=os.path.join(os.path.dirname(__file__),'..',\
                        'sample_eigenbasis_data',name)
    print('Loading eigenpairs from "%s"...'%filepath)
    
    f=open(filepath,'rb')
    
    E=pickle.load(f); f.close()
    
    if isinstance(E,dict):
        eigpairs={}
        for eigval in E: eigpairs[eigval-1]=E[eigval]
        return eigpairs
    
    eigpairs=dict([(E.axes[0][i],E[i]) for i in range(len(E))])
    
    return eigpairs

def planewave(qx,qy,x,y,x0=0,y0=0,phi0=0):
    
    return np.sin(qx*(x-x0)+qy*(y-y0)+phi0)

def build_rect_eigpairs(Lx=12,Rx=10,Ry=10,Nx=150,Ly=12,Nqmax=None):
    
    Rxmin,Rxmax=-Rx/2,+Rx/2
    Rymin,Rymax=-Ry/2,+Ry/2
    
    xs=np.linspace(-Lx/2,+Lx/2,Nx)
    Ny=int(Ly/Lx*Nx)
    ys=np.linspace(-Ly/2,+Ly/2,Ny)
    
    T=Timer()
    print('Generating eigenpairs on x,y=[-%s:+%s:%s],[-%s:+%s:%s]'%(Lx/2,Lx/2,Nx,Ly/2,Ly/2,Ny))
    
    global eigpairs
    global eigmultiplicity
    
    yv,xv = np.meshgrid(ys,xs)
    eigpairs = {}
    eigmultiplicity = {}
    
    Nqsx=int(Nx*Rx/Lx/2) #No use in including eigenfunctions with even greater periodicity
    Nqsy=int(Ny*Ry/Ly/2) #No use in including eigenfunctions with even greater periodicity
        
    
    q0x=np.pi/Rx #This is for particle in box (allowed wavelength is n*2*L)
    q0y=np.pi/Ry #This is for particle in box (allowed wavelength is n*2*L)
    
    qxs=np.arange(Nqsx+1)*q0x
    qys=np.arange(Nqsy+1)*q0y
    pairs=list(product(qxs,qys))
    eigvals=[qx**2+qy**2 for qx,qy in pairs]
    eigvals,pairs=zip(*sorted(zip(eigvals,pairs)))
    
    for eigval,pair in zip(eigvals[:Nqmax],\
                           pairs[:Nqmax]):
        if eigval==0: continue
        qx,qy=pair
        
        pw1=planewave(qx,0,xv,yv,\
                      x0=Rxmin,y0=Rymin,\
                      phi0=np.pi/2)
        pw2=planewave(0,qy,xv,yv,\
                      x0=Rxmin,y0=Rymin,\
                      phi0=np.pi/2)
        pw = AWA(pw1*pw2, axes = [xs,ys])

        pw[(xv<Rxmin)]=0
        pw[(xv>Rxmax)]=0
        pw[(yv<Rymin)]=0
        pw[(yv>Rymax)]=0
        pw-=np.mean(pw) #This is to ensure charge neutrality
        
        while eigval in eigpairs: eigval+=1e-8
        eigpairs[eigval]=normalize(pw)
    
    T()
            
    return eigpairs

def build_ribbon_eigpairs(Lx=12,Rx=10,Nx=150,Ly=12,Nqmax=None,qys=None):
    
    Rxmin,Rxmax=-Rx/2,+Rx/2
    
    xs=np.linspace(-Lx/2,+Lx/2,Nx)
    Ny=int(Ly/Lx*Nx)
    dy=Ly/Ny
    ys=np.arange(Ny)*dy
    T=Timer()
    print('Generating eigenpairs on x,y=[-%s:+%s:%s],[-%s:+%s:%s]'%(Lx/2,Lx/2,Nx,Ly/2,Ly/2,Ny))
    
    global eigpairs
    global eigmultiplicity
    
    yv,xv = np.meshgrid(ys,xs)
    eigpairs = {}
    eigmultiplicity = {}
    
    Nqsx=int(Nx*Rx/Lx/2) #No use in including eigenfunctions with even greater periodicity
    Nqsy=int(Ny/2) #No use in including eigenfunctions with even greater periodicity
    
    #make sure we have an even number of Nqsx and Nqsy
    if Nqsx%2: Nqsx+=1
    if Nqsy%2: Nqsy+=1
    
    q0x=np.pi/Rx #This is for particle in box (allowed wavelength is n*2*L)
    qxs=np.arange(0,Nqsx+1)*q0x
    
    if qys is None:
        q0y=2*np.pi/Ly #This is for periodic bcs (allowed wavelength is n*L)
        qys=np.arange(0,Nqsy+1)*q0y
        
    pairs=list(product(qxs,qys))
    eigvals=[qx**2+qy**2 for qx,qy in pairs]
    eigvals,pairs=zip(*sorted(zip(eigvals,pairs)))
    
    for eigval,pair in zip(eigvals[:Nqmax],\
                           pairs[:Nqmax]):
        
        #We cannot admit a constant potential, no charge neutrality
        if eigval==0: continue
        qx,qy=pair
        
        #First the cos*sin wave
        pw1=planewave(qx,0,xv,yv,\
                      x0=Rxmin,y0=ys.min(),\
                      phi0=np.pi/2)
        pw2=planewave(0,qy,xv,yv,\
                      x0=Rxmin,y0=ys.min(),\
                      phi0=0)
        pw = AWA(pw1*pw2, axes = [xs,ys])

        pw[(xv<Rxmin)]=0
        pw[(xv>Rxmax)]=0
        pw-=np.mean(pw) #This is to ensure charge neutrality
        
        if pw.any():
            while eigval in eigpairs: eigval+=1e-8
            eigpairs[eigval]=normalize(pw)
        
        #Second the cos*cos wave
        pw2=planewave(0,qy,xv,yv,\
                      x0=Rxmin,y0=-Ly/2,\
                      phi0=np.pi/2)
        pw = AWA(pw1*pw2, axes = [xs,ys])

        pw[(xv<Rxmin)]=0
        pw[(xv>Rxmax)]=0
        pw-=np.mean(pw) #This is to ensure charge neutrality
        
        if pw.any():
            while eigval in eigpairs: eigval+=1e-8
            eigpairs[eigval]=normalize(pw)
    
    T()
    
    return eigpairs

def build_homog_eigpairs(Lx=10,Nx=100,Ly=10,Nqmax=None):
    
    dx=Lx/Nx
    xs=np.arange(Nx)*dx; xs-=np.mean(xs)
    Ny=int(Ly/Lx*Nx)
    ys=np.arange(Ny)*dx; ys-=np.mean(ys)
    
    T=Timer()
    print('Generating eigenpairs on x,y=[-%s:+%s:%s],[-%s:+%s:%s]'%(Lx/2,Lx/2,Nx,Ly/2,Ly/2,Ny))
    
    global eigpairs
    global eigmultiplicity
    
    yv,xv = np.meshgrid(ys,xs)
    eigpairs = {}
    eigmultiplicity = {}
    
    Nqsx=int(Nx/4) #No use in including eigenfunctions with even greater periodicity
    Nqsy=int(Ny/4) #No use in including eigenfunctions with even greater periodicity
        
    
    q0x=2*np.pi/Lx #This is for periodic bcs (allowed wavelength is n*L)
    q0y=2*np.pi/Ly #This is for periodic bcs (allowed wavelength is n*L)
    
    qxs=np.arange(Nqsx+1)*q0x
    qys=np.arange(Nqsy+1)*q0y
    qpairs=list(product(qxs,qys))
    eigvals=[qx**2+qy**2 for qx,qy in qpairs]
    eigvals,qpairs=zip(*sorted(zip(eigvals,qpairs)))
    eigvals=list(eigvals)[:Nqmax//4] #factor of 4 is because we entertain a 4-fold degeneracy
    qpairs=list(qpairs)[:Nqmax//4]
    
    for eigval,qpair in zip(eigvals,qpairs):
        
        #We cannot admit a constant potential, no charge neutrality
        if eigval==0: continue
        qx,qy=qpair
        
        #First the cos*sin wave
        pwxs=planewave(qx,0,xv,yv,\
                      x0=np.min(xs),y0=np.min(ys),\
                      phi0=0)
        pwxc=planewave(qx,0,xv,yv,\
                      x0=np.min(xs),y0=np.min(ys),\
                      phi0=np.pi/2)
        pwys=planewave(0,qy,xv,yv,\
                      x0=np.min(xs),y0=np.min(ys),\
                      phi0=0)
        pwyc=planewave(0,qy,xv,yv,\
                      x0=np.min(xs),y0=np.min(ys),\
                      phi0=np.pi/2)
            
        for pw in [pwxc*pwyc,pwxs*pwyc,pwxc*pwys,pwxs*pwys]:
            
            if not pw.any(): continue
            
            pw = AWA(pw, axes = [xs,ys])
            
            while eigval in eigpairs: eigval+=1e-8
            eigpairs[eigval]=normalize(pw)
    
    T()
            
    return eigpairs


def Laplacian_periodic(functions,Lx,Ly,sigma=1):
    """
    Apply the position-dependent laplace operator to a list of
    (assumed) 2D periodic `functions` on a mesh of size `Lx,Ly`.
    Position-dependent conductivity `sigma` should be specified
    as an array of shape matching each function in `functions`.
    
    For some extensions to a spatially varying derivative:
        "Algorithm 3" of the great Dr. Steven G. Johnson
        https://math.mit.edu/~stevenj/fft-deriv.pdf
    """
    
    #Make sure we have a list of functions
    functions=np.array(functions)
    if functions.ndim==2:
        functions=functions.reshape((1,)+functions.shape)
    
    Nx,Ny=functions[0].shape
    dx,dy=Lx/Nx,Ly/Ny
    fx=np.fft.fftfreq(Nx,d=dx).reshape((1,Nx,1))
    fy=np.fft.fftfreq(Ny,d=dy).reshape((1,1,Ny))
    indxmin=Nx//2; indymin=Ny//2

    sigma=np.array(sigma)
    sigma.resize((1,)+sigma.shape)
    while sigma.ndim<3: sigma.resize(sigma.shape+(1,))
        
    S=sigma
    Savx=np.mean(S,axis=1)
    Savy=np.mean(S,axis=2)

    print('Computing Laplacian...')
    T=Timer()

    u=np.array(functions)
    Uxs=np.fft.fft(u,axis=1); Uxs_fmin=Uxs[:,indxmin,:]
    Uys=np.fft.fft(u,axis=2); Uys_fmin=Uys[:,:,indymin]

    Vx = np.fft.ifft(2*np.pi*1j*fx*Uxs,axis=1)
    Vy = np.fft.ifft(2*np.pi*1j*fy*Uys,axis=2)
    del Uxs,Uys
    if not np.iscomplexobj(S):
        Vx=np.real(Vx)
        Vy=np.real(Vy)

    Vxs=np.fft.fft(S*Vx,axis=-2)
    Vys=np.fft.fft(S*Vy,axis=-1)
    del Vx,Vy

    # if N even
    # This step preserves self-adjointness of L operator, according to Dr. Johnson
    # The reasoning is above my paygrade
    if not Nx%2: Vxs[:,indxmin,:]= -Savx*(np.pi/dx)**2*Uxs_fmin
    if not Ny%2: Vys[:,:,indymin]= -Savy*(np.pi/dy)**2*Uys_fmin

    d2udx2 = np.fft.ifft(2*np.pi*1j*fx*Vxs,axis=1)
    d2udy2 = np.fft.ifft(2*np.pi*1j*fy*Vys,axis=2)
    del Vxs,Vys
    if not np.iscomplexobj(S):
        d2udx2=np.real(d2udx2)
        d2udy2=np.real(d2udy2)
    
    Lu=-(d2udx2+d2udy2) #This gives us positive eigenvalues
    T()
    
    return Lu

def build_inhomog_eigpairs(Lx=10,Nx=100,Ly=10,\
                              Nqmax=None,dsigma=0,\
                              multiprocess=False):
    """
    Build an eigenbasis for a rectangular periodic space of
    size `Lx` x `Ly` with pixel density given by `Nx`,
    modulated by inhomogeneous conductivity `sigma`.
    The generated eigenbasis diagonalizes the spatially
    inhomgeneous but periodic Laplacian.
    

    Parameters
    ----------
    Lx : number, optional
        Size in x direction. The default is 10.
    Nx : `int`, optional
        Number of pixels along x direction. The default is 100.
    Ly : number, optional
        Size in y direction. The default is 10.
    Nqmax : `int`, optional
        Maximum size of eigenbasis. The default is None.
    dsigma: `np.ndarray` or has `__call__` method, optional
        Complex array or callable function of x,y
        describing position-dependent part of conductivity.
        Since background conductivity is assumed unity,
        `dsigma` should be small by comparison.
        The default is 0.
    multiprocess : `bool`, optional
        Use multiprocessing module? The default is False.
        Not yet implemented, could use it for Laplacian.

    Returns
    -------
    eigpairs : `dict`
        Dictionary of complex eigenvalue / eigenfunction array
        pairs that diagonalize the position-dependent Laplacian
        operator with conductivity `sigma`.
        
    TODO: break-out the `L_mn` and `dL_mn` matrices so they can
          be re-used to build other eigenbases if the perturbation
          is scaled by an arbitrary constant.
    """
    
    #Build eigpairs for homogeneous problem
    eigpairs0=build_homog_eigpairs(Lx=Lx,Nx=Nx,Ly=Ly,Nqmax=Nqmax)
    eigvals0=list(eigpairs0.keys())
    eigfuncs0=list(eigpairs0.values())
    Us0=np.array([eigfunc0.ravel() for eigfunc0 in eigfuncs0]).T
    Xs,Ys=eigfuncs0[0].axis_grids; xs,ys=eigfuncs0[0].axes
    eigfuncs0=[np.array(eigfunc) for eigfunc in eigfuncs0] #We don't want AWA from here
    
    #Figure out the provided conductivity
    if hasattr(dsigma,'__call__'): dsigma=dsigma(Xs,Ys)
    
    #Build matrix representation for Laplacian and diagonalize it
    # Split into two parts:
    # 1) L0_mn: for spatially homogeneous part of conductivity (unity);
    #           This is already diagonal and elements are known
    #           anaylytically for plane waves (just q_n**2)
    # 2) dL_mn: for spatially varying part;
    #           This matrix should be small compared to L0_mn, and
    #           that way inperfections in numerical laplacian won't
    #           be too problematic
    dLeigfuncs0=Laplacian_periodic(eigfuncs0,Lx,Ly,sigma=dsigma)
    dL_mn=build_matrix(eigfuncs0,dLeigfuncs0) #matrix elements for peturbing conductivity
    L0_mn=np.diag(eigvals0)
    L_mn=L0_mn+dL_mn
    if np.iscomplexobj(dsigma): eig=np.linalg.eig #We can't assume `L_mn` is hermitian
    else: eig=np.linalg.eigh #We can certainly assume `L_mn` is hermitian
    eigvals,eigvecs=eig(L_mn)
    
    #Sort the eigenvectors & try to align vectors along real axis
    idx = eigvals.argsort() #This will sort by real value- do we want abs?
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    if np.iscomplexobj(dsigma): #disabled for now, shoudn't matter
        print('Aligning phase of eigenvectors')
        T=Timer()
        for i in range(len(eigvecs)):
            eigvecs[:,i]=align_to_reals(eigvecs[:,i])
        
        T()
    
    #Build eigenbasis from specified linear combinations of homogeneous basis
    Us = Us0 @ eigvecs #Hit `Us0` with all our column vectors
    eigfuncs=[AWA(u.reshape(eigfuncs0[0].shape),\
                  axes=[xs,ys],axis_names=['x','y']) \
              for u in Us.T]
    
    eigpairs=dict(zip(eigvals,eigfuncs))
    
    return eigpairs


# -*- coding: utf-8 -*-

