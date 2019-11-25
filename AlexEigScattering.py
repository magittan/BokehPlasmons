#import Plasmon_Modeling as PM
#from dolfin import *
#from fenics import *
#from mshr import *
#TODO: check units of eigenvalues, with proper mesh spacing in Helmholtz solver
import BasisChangeTest as BCT
import Plasmon_Modeling as PM
import CoulombKernel as CK
import os,time, h5py
basedir=os.path.dirname(__file__)

from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA
from scipy import special as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import multiprocessing as mp
from itertools import product
#from numba import cuda
#%matplotlib notebook

def Progress(i,L,last):
    next = last+10
    percent = 100*i/L
    if percent >= next:
        print('{}% complete...'.format(next))
        return next
    else:
        return last

def Calc(psi,psi_star):
    return np.sum(psi_star*myQC(psi))

# Basis change functions
def ChangeBasis(to_eigenbasis,from_eigenbasis,xs,ys):
    to_nn = range(to_eigenbasis.shape[0])
    from_nn = range(from_eigenbasis.shape[0])
    U = np.zeros((len(to_nn), len(from_nn)))
    last, N = 0, len(to_nn)*len(from_nn)
    for to_n in to_nn:
        for from_n in from_nn:
            from_ef = to_eigenbasis[to_n]
            to_ef = from_eigenbasis[from_n]
            dot_prod = np.asarray(from_ef*to_ef)
            sum = np.trapz(np.trapz(dot_prod,ys),xs)
            #sum = nquad(lambda x,y: IntFun(dot_prod,x,y),[[-1,1],[-1,1]])
            U[to_n,from_n] = sum
            #last = Progress(to_n*from_n, N, last)
    return U

"""
@cuda.jit
def ChangeBasisGPU(U, to_eigenbasis,from_eigenbasis,xs,ys):
    row, col = cuda.grid(2)
    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]
    sx, sy = to_eigenbasis[0].shape
    if row < U.shape[0] and col < U.shape[1]:
        sum = 0.
        to_ef = to_eigenbasis[row]
        from_ef = from_eigenbasis[col]
        for i in range(from_ef.shape[0]):
            for j in range(from_ef.shape[1]):
                sum += from_ef[i,j]*to_ef[i,j]*dx*dy
        cuda.syncthreads()
        U[row,col] = sum
        #U.append(row)
    #return U
"""

class BesselGenerator(object):
    """
    Generator of Bessel functions placed at arbitrary "center" position
    within an xy mesh space.  Works by generating a bessel function on
    a much larger xy mesh and then translating/truncating to the
    original mesh.  Nothing fancy, but good for performance.
    """

    def __init__(self,q=20,\
                 xs=np.linspace(-1,1,101),\
                 ys=np.linspace(-1,1,101)):

        #Bookkeeping of the coordinate mesh
        self.xs,self.ys=xs,ys
        self.shape=(len(xs),len(ys))
        self.midx=self.xs[self.shape[0]//2]
        self.midy=self.ys[self.shape[1]//2]
        self.dx=np.max(self.xs)-np.min(self.xs)
        self.dy=np.max(self.ys)-np.min(self.ys)

        #Make a mesh grid twice bigger in each direction
        bigshape=[2*N-1 for N in self.shape]
        xs2grid,ys2grid=np.ogrid[self.midx-self.dx:self.midx+self.dx:bigshape[0]*1j,
                                 self.midy-self.dy:self.midy+self.dy:bigshape[1]*1j]
        self.xs2=xs2grid.squeeze()
        self.ys2=ys2grid.squeeze()

        N_tip_eigenbasis = 10
        J_zeros = sp.jn_zeros(0,N_tip_eigenbasis)
        rs2 = np.sqrt(xs2grid**2+ys2grid**2)
        self.Jbasis = np.array([(2/sp.jv(1,J_zeros[n]))*sp.jv(0,J_zeros[n]*rs2) for n in range(N_tip_eigenbasis)])

        self.Jv = np.random.randint(0,high=5,size=N_tip_eigenbasis)
        print('Excitation vector (in basis of J(n), n in [0,{}]):\n\t{}'.format(N_tip_eigenbasis,self.Jv))

        self.Jm = np.array([u*self.Jbasis[i] for i,u in enumerate(self.Jv)])
        self.bigJ=np.sum(self.Jm,axis=0)

    def GetTipEigenbasis(self,x0,y0):
        indx=np.argmin(np.abs(x0-self.xs2))
        indy=np.argmin(np.abs(y0-self.ys2))
        shiftx=self.shape[0]+indx
        shifty=self.shape[1]+indy

        tip_eigenbasis = []
        for u in self.Jbasis:
            newJ=np.roll(np.roll(u,shiftx,axis=0),\
                         shifty,axis=1)
            output = newJ[self.shape[0]//2:(3*self.shape[0])//2,\
                        self.shape[1]//2:(3*self.shape[1])//2]
            tip_eigenbasis.append(output)
        return np.array(tip_eigenbasis)

    def __call__(self,x0,y0):
        indx=np.argmin(np.abs(x0-self.xs2))
        indy=np.argmin(np.abs(y0-self.ys2))
        shiftx=self.shape[0]+indx
        shifty=self.shape[1]+indy

        newJ=np.roll(np.roll(self.bigJ,shiftx,axis=0),\
                     shifty,axis=1)
        output = newJ[self.shape[0]//2:(3*self.shape[0])//2,\
                    self.shape[1]//2:(3*self.shape[1])//2]
        return output

class SampleResponse(object):
    """
        Generator of sample response based on an input collection of
        eigenpairs (dictionary of eigenvalues + eigenfunctions).

        Can output a whole set of sample response functions from
        an input set of excitation functions.

        Usage will be something like:

        >>> from BasovPlasmons import AlexEigScattering as AES
        >>> eigpairs = AES.load_eigpairs()
        >>> xs=ys=np.linspace(-1,1,101)
        >>> Jmaker=AES.BesselGenerator(q=20,xs=xs,ys=ys) #To provide excitations
        >>> excitations=[Jmaker(x0,y0) for x0,y0 in zip(some_xs,some_ys)]
        >>> SR=AES.SampleResponse(eigpairs,E=1400,N=125)
        >>> responses=SR(excitations)
    """

    def __init__(self,eigpairs,E,N=100,debug=True):

        eigvals=list(eigpairs.keys())
        eigfuncs=list(eigpairs.values())
        self.xs,self.ys=eigfuncs[0].axes

        self.eigfuncs=AWA(eigfuncs,\
                          axes=[eigvals,self.xs,self.ys]).sort_by_axes()
        self.eigvals=self.eigfuncs.axes[0]

        self.phishape=self.eigfuncs[0].shape
        self.E=E
        self.N=N

        #Aim eigenfunctions
        if debug: print('Setting Energy')
        self._SetEnergy(E)
        if debug: print('Setting Sigma')
        self._SetSigma(10,10)
        if debug: print('Setting Kernel')
        self._SetCoulombKernel()
        if debug: print('Setting Scattering Matrix')
        self._SetScatteringMatrix()

    def _SetEnergy(self,E):
        """
            TODO: check energy units and sqrt eigenvals
        """
        index=np.argmin(np.abs(self.eigvals-E))
        ind1=np.max([index-self.N//2,0])
        ind2=ind1+self.N

        self.use_eigfuncs=self.eigfuncs[ind1:ind2]
        self.Phis=np.matrix([eigfunc.ravel() for eigfunc in self.use_eigfuncs])

        self.use_eigvals=self.eigvals[ind1:ind2]
        self.Q = np.diag(self.use_eigvals)

    def _SetSigma(self,L,lamb):
        self.sigma = PM.S()
        self.sigma.set_sigma_values(lamb, L)
        sigma_tilde = self.sigma.get_sigma_values()[0]+1j*self.sigma.get_sigma_values()[1]
        self.alpha = -1j*sigma_tilde/np.abs(sigma_tilde)

    def _SetCoulombKernel(self):
        self.V_nm = np.zeros([len(self.use_eigvals), len(self.use_eigvals)])
        eigfuncs = self.use_eigfuncs
        kern_func = lambda x,y: 1/np.sqrt(x**2+y**2+1e-4)
        global myQC
        myQC=numrec.QuickConvolver(size=(100,100),kernel_function=kern_func,shape=eigfuncs[0].shape,pad_by=.5,pad_with=0)
        self.myQC=numrec.QuickConvolver(size=(100,100),kernel_function=kern_func,shape=eigfuncs[0].shape,pad_by=.5,pad_with=0)

        #start = time.time()
        p = mp.Pool(5)
        self.V_nm = np.array(p.starmap(Calc,product(eigfuncs,eigfuncs))).reshape((self.N,self.N))
        #V_nm_from_mp = np.array(p.starmap(Calc,product(eigfuncs,eigfuncs))).reshape((self.N,self.N))
        #print("MP time: {}".format(time.time()-start))

        """
        start = time.time()
        for n in range(0,self.N):
            for m in range(0,self.N):
                psi = eigfuncs[m]
                psi_star = eigfuncs[n]
                self.V_nm[n,m]=np.sum(psi_star*self.myQC(psi))

        print("Non mp time: {}".format(time.time()-start))
        print(np.max(V_nm_from_mp-self.V_nm))
        """

    def _SetScatteringMatrix(self):
        self.D = self.E*np.linalg.inv(self.E*np.identity(self.Q.shape[0]) - self.alpha*self.Q.dot(self.V_nm))

    def GetRAlphaBeta(self, tip_eigenbasis):
        #start = time.time()
        U = BCT.ChangeBasisGPU(tip_eigenbasis,self.use_eigfuncs,self.xs,self.ys)
        #GPU_elapsed = time.time()-start
        #print("Basis Change (tip to sample) with GPU: {} seconds".format(GPU_elapsed))
        #print("Speedup: {}".format(unopt_elapsed/GPU_elapsed))
        #U = ChangeBasis(tip_eigenbasis, self.use_eigfuncs, self.xs, self.ys)
        #plt.figure();plt.imshow(U);plt.show();
        #U_inv = np.linalg.inv(U)
        U_inv = U.T
        print('U shape: {}\nU_inv shape: {}'.format(U.shape,U_inv.shape))
        result = np.dot(U.T,np.dot(self.D,U))
        time.sleep(1)
        return result

    def __call__(self,excitations,U,tip_eigenbasis):

        if np.array(excitations).ndim==2: excitations=[excitations]
        Exc=np.array([exc.ravel() for exc in excitations])
        tip_eb=np.matrix([eigfunc.ravel() for eigfunc in tip_eigenbasis])
        projected_result=np.dot(tip_eb.T,
                            np.dot(U,\
                                np.dot(self.D,\
                                    np.dot(self.Phis,Exc.T))))
        #plt.figure();plt.imshow(np.abs(result.reshape(101,101)));plt.show()

        #These are all the matrices that get multiplied, take a look that shapes work...
        #print([item.shape for item in [self.Phis.T,self.D,self.Phis,Exc.T]])
        #result is in form of column vectors
        #turn into row vectors then reshape
        result=np.dot(self.Phis.T,\
                       np.dot(self.D,\
                             np.dot(self.Phis,Exc.T)))
        result=np.array(result).T.reshape((len(excitations),)+self.phishape)
        projected_result=np.array(projected_result).T.reshape((len(excitations),)+self.phishape)
        return AWA(result,axes=[None,self.xs,self.ys]).squeeze(), AWA(projected_result,axes=[None,self.xs,self.ys]).squeeze()

#--- Example functions
def load_eigpairs():
    """Are eigenvalues correct?? They seem to be for Laplace operator on mesh of 0-101 range.

    Normalization by sum always ensures that integration will be like summing, which is
    much simpler than keeping track of dx, dy..."""

    global eigpairs
    eigpairs = dict()

    path=os.path.join(basedir,"UnitSquareMesh_100x100_1000_eigenbasis.h5")

    with h5py.File(path,'r') as f:
        for key in list(f.keys()):
            eigfunc=np.array(f.get(key))
            eigfunc/=np.sqrt(np.sum(np.abs(eigfunc)**2))
            eigpairs[float(key)] = AWA(eigfunc,\
                                       axes=[np.linspace(-1,1,eigfunc.shape[0]),\
                                             np.linspace(-1,1,eigfunc.shape[1])])

    #return eigpairs

def scan_qs_at_E(qs=np.linspace(5,30,200),\
                 E=2000*np.exp(1j*2*np.pi*5e-2),\
                 N=500,y0=0):
    """Scan a bunch of excitation wave vectors for a fixed choice of energy."""

    #Scan qs
    global Responder,Jmaker

    Responder=SampleResponse(eigpairs,E=E,N=N)
    xs,ys=Responder.xs,Responder.ys

    y0=y0
    output=np.zeros((len(qs),len(xs)),dtype=complex)
    last = 0
    for i,q in enumerate(qs):

        Jmaker=BesselGenerator(q,xs=xs,ys=ys)
        excitations=[Jmaker(x0,y0) for x0 in xs]

        responses=Responder(excitations)
        output[i]=[response.cslice[x0,y0] for response,x0 in zip(responses,xs)]
        last = Progress(i,len(qs),last)

    output=AWA(output,axes=[qs,xs],axis_names=['$q$','$X$'])
    plt.figure();np.abs(output).plot()
    plt.title('$q$-sweep at E=%1.2f'%np.abs(E))
    plt.tight_layout()

    return output

def scan_source_at_q_and_E(q=20,\
                           E=2000*np.exp(1j*2*np.pi*5e-2),\
                           N=100):
    """Raster scan a bunch of bessel waves over the sample and look at the
    sample response at each excitation point.  Computes sample response in
    parallel for excitations placed at each x-position; could  try to
    compute all responses in parallel limited only by CPU memory..."""

    global Responder,Jmaker

    Responder=SampleResponse(eigpairs,E=E,N=N)
    xs,ys=Responder.xs,Responder.ys
    Jmaker=BesselGenerator(q,xs=xs,ys=ys)

    output=np.zeros((len(xs),len(ys)),dtype=complex)
    projected_output=np.zeros((len(xs),len(ys)),dtype=complex)
    last = 0
    to_efs = Jmaker.smallJbasis
    from_efs = Responder.use_eigfuncs
    U = ChangeBasis(to_efs,from_efs,xs,ys)
    for i,x0 in enumerate(xs):
        #excitations=[Jmaker(x0,y0) for y0 in ys]
        #responses=Responder(excitations,U,to_efs)
        for j,y0 in enumerate(ys):
            #print(i,j)
            excitation,_=Jmaker(x0,y0)
            response, projected_response=Responder(excitation,U,to_efs)
            output[i,j]=response.cslice[x0,y0]
            projected_output[i,j]=projected_response.cslice[x0,y0]
            last = Progress(i,len(xs),last)
    output=AWA(output,axes=[xs,ys],axis_names=['$X$','$Y$'])
    projected_output=AWA(projected_output,axes=[xs,ys],axis_names=['$X$','$Y$'])
    plt.figure();np.abs(output).plot()
    plt.title('Response at scanned positions')
    plt.tight_layout()
    plt.figure();np.abs(projected_output).plot()
    plt.title('Projected response at scanned positions')
    plt.tight_layout()
    plt.show()

    return output

def TestScatteringBasisChange(q=20,\
                           E=2000*np.exp(1j*2*np.pi*5e-2),\
                           N=100):

    global Responder,Jmaker

    Responder=SampleResponse(eigpairs,E=E,N=N)
    xs,ys=Responder.xs,Responder.ys
    Jmaker=BesselGenerator(q,xs=xs,ys=ys)

    last = 0
    for i,x0 in enumerate(xs):
        for j,y0 in enumerate(ys):
            print(i,j)
            start = time.time()
            tip_eigenbasis=Jmaker.GetTipEigenbasis(x0,y0)
            Responder.GetRAlphaBeta(tip_eigenbasis)
            elapsed = time.time()-start
            print('Time elapsed per loop: {} s'.format(elapsed))
            last = Progress(i,len(xs),last)

    return output
