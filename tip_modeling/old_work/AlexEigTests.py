# -*- coding: utf-8 -*-
from BokehPlasmons import tip_modeling as TM
import os,h5py,time
#import Plasmon_Modeling as PM
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy import special as sp
from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA
from itertools import product
from BokehPlasmons.tip_modeling import utils
import pickle


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

def build_ribbon_eigpairs(Lx=12,Rx=10,Nx=150,Ly=12,Nqmax=None,qys=None):
    
    Rxmin,Rxmax=-Rx/2,+Rx/2
    
    xs=np.linspace(-Lx/2,+Lx/2,Nx)
    Ny=int(Ly/Lx*Nx)
    dy=Ly/Ny
    ys=np.arange(Ny)*dy
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

def build_homog_eigpairs(Lx=10,Nx=100,Ly=10,Nqmax=None):
    
    dx=Lx/Nx
    xs=np.arange(Nx)*dx
    Ny=int(Ly/Lx*Nx)
    ys=np.arange(Ny)*dx
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
    pairs=list(product(qxs,qys))
    eigvals=[qx**2+qy**2 for qx,qy in pairs]
    eigvals,pairs=zip(*sorted(zip(eigvals,pairs)))
    
    for eigval,pair in zip(eigvals[:Nqmax],\
                           pairs[:Nqmax]):
        
        #We cannot admit a constant potential, no charge neutrality
        if eigval==0: continue
        qx,qy=pair
        
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
            eigpairs[eigval]=pw
            
    return eigpairs

def TestEigenbasisProjection(eigpairs,N=1000,\
                             src=None,y0=None,
                             qp=0,Qfactor=20,
                             plot=True,**kwargs):
    """This can test ringing artifacts in the projection of sources onto the eigenbasis."""
    
    
    global refl,R
    
    #The qp value etc. is immaterial to this test
    R=TM.SampleResponse(eigpairs,qp=qp,Qfactor=Qfactor,N=N,\
                         coulomb_shortcut=True,coulomb_bc='periodic')
        
    Nx=len(R.xs)
    if not y0: y0=np.mean(R.ys)
    
    if src is None:
        x0=np.mean(R.xs)
        X,Y=R.getXYGrids()
        src=TM.gaussianfield(X-x0,Y-y0,**kwargs)
    
    #We rely on a source evaluated in the center, then we roll it to preserve
    src-=np.mean(src)
    sources=[np.roll(src,i-Nx//2,axis=0) for i in range(len(R.xs))]
    sources=AWA(sources, axes=[R.xs,R.xs,R.ys],axis_names=[r'$x_0$','x','y'])
    
    print('Projecting to real space from eigenbasis...')
    projections=R.perfectly_reflect(sources)
    projections.adopt_axes(sources)
    
    deviations=AWA([np.sum((proj-src)**2)/np.sum(src**2)*100 for \
               proj,src in zip(projections,sources)],\
              axes=[R.xs],axis_names=[r'$x_0$'])
    
    print('Auto-projecting from eigenbasis...')
    refls=[R.get_reflection_coefficient(src).real for src in sources]
    refls=AWA(refls).squeeze(); refls.adopt_axes(deviations)
    autoprojections=AWA(refls,axes=[R.xs],axis_names=[r'$x_0$'])
    
    if plot:
        plt.figure(figsize=(18,4.5))
        
        plt.subplot(141)
        sources.cslice[:,:,y0].plot()
        plt.title(r'$V_\mathrm{exc}$',fontsize=22)
        plt.gca().set_aspect('equal')
        
        plt.subplot(142)
        projections.cslice[:,:,y0].plot()
        plt.title(r'$V_\mathrm{ind}$',fontsize=22)
        plt.gca().set_aspect('equal')
        
        plt.subplot(143)
        deviations.plot()
        plt.title(r'$V_\mathrm{exc}-V_\mathrm{ind}$ (%)',fontsize=22)
        
        plt.subplot(144)
        autoprojections.plot()
        plt.title('"Reflectance"')
        
        plt.tight_layout()
    
    return dict(sources=sources,projections=projections,\
                deviations=deviations,autoprojections=autoprojections)

def SweepDipoleOver(y=0,dipolez=1,\
                       qp=2*np.pi,Qfactor=10,
                       eigpairs=None,\
                       Lx=12,Rx=10,Ry=10,Nx=150,Ly=12,
                        Responder=None,beta=0,\
                            periodicx=False,\
                            **kwargs):
    
    if Responder is None:
        global TheResponder
        if eigpairs is None:
            eigpairs=build_rect_eigpairs(Lx=Lx,Rx=Rx,Ry=Ry,Nx=Nx,Ly=Ly)
        Responder=TM.SampleResponse(eigpairs,qp=qp,Qfactor=Qfactor,**kwargs)
        TheResponder=Responder
    
    X,Y=Responder.getXYGrids()
    xs=Responder.xs
    
    if periodicx:
        #evaluate at center then roll into position
        xmid=np.mean(xs)
        src=utils.dipole_field(X-xmid,Y-y,z=dipolez)
        excitations=[np.roll(src,i-len(xs)//2,axis=0) \
                     for i in range(len(xs))]
    else:
        excitations=[utils.dipole_field(X-x,Y-y,dipolez) for x in xs]
    
    responses=beta+Responder(excitations)*(1-beta)
    linecut=[response[i].cslice[y] for i,response in enumerate(responses)]
    linecut=AWA(linecut,axes=[xs])
    
    return dict(responses=responses,Responder=Responder,linecut=linecut,excitations=excitations)


def demo_SaiDomainBoundary(filename='Eigenbasis_SaiConductivity_xy200x200_NxNy200x200_Nq=1000.pickle',\
                           tipsize=50,lp_min=20,lp_max=200,Nlps=20,\
                           Qfactor=10,coulomb_shortcut=False,sample_response=None):
    
    global SR,TR
    if sample_response is None:
        eigpairs=load_eigpairs(filename)
        SR=TM.SampleResponse(eigpairs,qp=0,Qfactor=Qfactor,N=1000,\
                                coulomb_shortcut=coulomb_shortcut,coulomb_bc='periodic',\
                                coulomb_multiprocess=False)
    else: SR=sample_response
    
    #Because sample response is periodic, consistency requires tip response with periodic translators
    TR=TM.TipResponse(SR.xs,SR.ys,\
                      q=2*np.pi/tipsize,\
                      N_tip_eigenbasis=1,\
                      periodic=True)
    
    rasters=[]
    lps=np.linspace(lp_min,lp_max,Nlps)
    for i,lp in enumerate(lps):
        SR._set_reflection_matrix(2*np.pi/lp)
        rasters.append(SR.raster_scan(TR,ys=0)['R'])
        print('PROGRESS: %1.1f%% done'%((i+1)/len(lps)*100))
        
    return AWA(rasters,axes=[lps]+rasters[0].axes,\
               axis_names=[r'$\lambda_p$ (nm)','$x$ (nm)'])

