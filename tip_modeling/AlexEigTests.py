# -*- coding: utf-8 -*-
import AlexEigScattering as AES
import os,h5py,time
#import Plasmon_Modeling as PM
import multiprocessing as mp
import numpy as np
from scipy import special as sp
from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA
import matplotlib.pyplot as plt
from itertools import product
from Utils import Progress, load_eigpairs ,planewave
import pickle

"""
def load_eigpairs(name='UnitSquareMesh_100x100_1000_eigenbasis.h5'):

    filepath=os.path.join(os.path.dirname(__file__),'..',\
                        'sample_eigenbasis_data',name)
    print('Loading eigenpairs from "%s"...'%filepath)

    f=open(filepath,'rb')

    E=pickle.load(f); f.close()
    eigpairs=dict([(E.axes[0][i],E[i]) for i in range(len(E))])

    return eigpairs
"""

def build_rect_eigpairs(Lx=12,Rx=10,Ry=10,Nx=150,Ly=12,Nqmax=None):

    Rxmin,Rxmax=-Rx/2,+Rx/2
    Rymin,Rymax=-Ry/2,+Ry/2

    xs=np.linspace(-Lx/2,+Lx/2,Nx)
    Ny=int(Ly/Lx*Nx)
    ys=np.linspace(-Ly/2,+Ly/2,Ny)
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
        eigpairs[eigval]=pw


    return eigpairs

def build_ribbon_eigpairs(Lx=12,Rx=10,Nx=150,Ly=12,Nqmax=None):

    Rxmin,Rxmax=-Rx/2,+Rx/2

    xs=np.linspace(-Lx/2,+Lx/2,Nx)
    Ny=int(Ly/Lx*Nx)
    ys=np.linspace(-Ly/2,+Ly/2,Ny)
    print('Generating eigenpairs on x,y=[-%s:+%s:%s],[-%s:+%s:%s]'%(Lx/2,Lx/2,Nx,Ly/2,Ly/2,Ny))

    global eigpairs
    global eigmultiplicity

    yv,xv = np.meshgrid(ys,xs)
    eigpairs = {}
    eigmultiplicity = {}

    Nqsx=int(Nx*Rx/Lx/2) #No use in including eigenfunctions with even greater periodicity
    Nqsy=int(Ny/2) #No use in including eigenfunctions with even greater periodicity

    q0x=np.pi/Rx #This is for particle in box (allowed wavelength is n*2*L)
    q0y=2*np.pi/Ly #This is for periodic bcs (allowed wavelength is n*L)

    qxs=np.arange(0,Nqsx+1)*q0x
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
                      x0=Rxmin,y0=-Ly/2,\
                      phi0=np.pi/2)
        pw2=planewave(0,qy,xv,yv,\
                      x0=Rxmin,y0=-Ly/2,\
                      phi0=0)
        pw = AWA(pw1*pw2, axes = [xs,ys])

        pw[(xv<Rxmin)]=0
        pw[(xv>Rxmax)]=0
        pw-=np.mean(pw) #This is to ensure charge neutrality

        if pw.any():
            while eigval in eigpairs: eigval+=1e-8
            eigpairs[eigval]=pw

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
            eigpairs[eigval]=pw

    return eigpairs

def RasterScanTip(xs=None,ys=None,\
                    qtip=2*np.pi/.15,qp=2*np.pi/.15,
                    N_tip_eigenbasis = 1,\
                    Responder=None,eigpairs=None,\
                    Qfactor=50,\
                    eigmultiplicity={},\
                     N_sample_eigenbasis=100,\
                    coulomb_shortcut=False):

    global TipEigenbasis
    if eigpairs is None:
        eigpairs = build_rect_eigpairs()

    if Responder is None:
        Responder = AES.SampleResponse(eigpairs,qp=qp,Qfactor=Qfactor,\
                                       eigmultiplicity=eigmultiplicity,\
                                       N=N_sample_eigenbasis,\
                                       coulomb_shortcut=coulomb_shortcut)

    if xs is None: xs=Responder.xs
    elif not hasattr(xs,'__len__'): xs=[xs]

    if ys is None: ys=Responder.ys
    elif not hasattr(ys,'__len__'): ys=[ys]

    TipEigenbasis = AES.TipResponse(Responder.xs,Responder.ys,\
                                    q=qtip,N_tip_eigenbasis=N_tip_eigenbasis)

    tip_eigenpoles = np.diag((2-.1j)*np.exp(np.arange(N_tip_eigenbasis)))
    tip_eigenresidues = 1+np.arange(N_tip_eigenbasis) #np.exp(np.arange(N_tip_eigenbasis))

    Ps=np.zeros((len(xs),len(ys)),dtype=np.complex)
    Rs=np.zeros((len(xs),len(ys)),dtype=np.complex)
    last = 0
    print("Raster scanning tip...")
    start = time.time()
    for i,x0 in enumerate(xs):
        for j,y0 in enumerate(ys):

            tip_eigenbasis = TipEigenbasis(x0,y0)
            Rsample = Responder.GetReflectionCoefficient(tip_eigenbasis)

            P_sample=np.sum(np.linalg.inv(tip_eigenpoles-Rsample).dot(tip_eigenresidues))
            P_0=np.sum(np.linalg.inv(tip_eigenpoles).dot(tip_eigenresidues))

            Ps[i,j] = P_sample-P_0
            Rs[i,j] = np.sum(np.diag(Rsample))/N_tip_eigenbasis

            last = Progress(i,len(xs),last)

            #if i==0 and j==0:
            #    plt.figure()
            #    plt.imshow(np.abs(R_alphabeta))
            #    plt.show()
    print("\tTime elapsed:{}".format(time.time()-start))
    return {'P':AWA(Ps,axes=[xs,ys],axis_names=['x','y']).squeeze(),\
            'R':AWA(Rs,axes=[xs,ys],axis_names=['x','y']).squeeze(),\
            'Responder':Responder}

def CompareGeneratedLoadedRecEigenbasis(wl=.15,Qfactor=25,N_tip_eigenbasis=1,\
                                        coulomb_shortcut=True):

    qtip=2*np.pi/wl
    qp=2*np.pi/wl

    eigpairs_gen=build_rect_eigpairs(Lx=1,Rx=1,Ry=1,Nx=101,Ly=1,Nmax=1000)
    eigpairs_loaded=load_eigpairs(name='UnitSquareMesh_101x101x2000_Neumann_eigenbasis.pickle')

    d_gen=RasterScanTip(eigpairs_gen,qtip=qtip,qp=qp,Qfactor=Qfactor,\
                        N_tip_eigenbasis=N_tip_eigenbasis,\
                        coulomb_shortcut=coulomb_shortcut)
    d_loaded=RasterScanTip(eigpairs_loaded,qtip=qtip,qp=qp,Qfactor=Qfactor,\
                           N_tip_eigenbasis=N_tip_eigenbasis,\
                           coulomb_shortcut=coulomb_shortcut)

    return dict(loaded=d_loaded,gen=d_gen)

def LinescanSweepingQp(wltip=1,N_tip_eigenbasis=1,
                       Nx=150,Lx=12,Ly=2,
                       wlmax=5,wlmin=.1,Nqps=50,Qfactor=25,
                       N_sample_eigenbasis=1000,
                       coulomb_shortcut=True,\
                       coulomb_bc='closed'):

    global Responder
    qtip=2*np.pi/wltip
    qps=2*np.pi/np.linspace(wlmax,wlmin,Nqps)

    eigpairs=build_ribbon_eigpairs(Nx=Nx,Lx=Lx,Ly=Ly,Nqmax=None)
    Responder=AES.SampleResponse(eigpairs,qp=qtip,N=N_sample_eigenbasis,\
                                 Qfactor=Qfactor,\
                                 coulomb_shortcut=coulomb_shortcut,\
                                 coulomb_bc=coulomb_bc)
    xs=Responder.xs

    Rs=[]; Ps=[]
    for i,qp in enumerate(qps):
        print('Computing for qp=%s...'%qp)

        #Just recompute reflection matrix, but don't re-tune eigenfunctions
        #without `diag=True`, off diagonal components from Coulomb matrix cause negative imaginary part

        Responder._SetReflectionMatrix(qp,diag=True)

        linescan=RasterScanTip(xs=xs,ys=0,qtip=qtip,\
                               N_tip_eigenbasis = N_tip_eigenbasis,\
                               Responder=Responder)
        Rs.append(linescan['R'])
        Ps.append(linescan['P'])

    Rs=AWA(Rs,axes=[2*np.pi/qps,xs],axis_names=[r'$\lambda_p$','x'])
    Ps=AWA(Ps); Ps.adopt_axes(Rs)

    return dict(P=Ps,R=Rs,Responder=Responder)

RasterScanTip()
