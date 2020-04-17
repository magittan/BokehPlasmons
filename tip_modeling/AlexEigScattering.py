import os,h5py,time
#import Plasmon_Modeling as PM
import multiprocessing as mp
import numpy as np
from scipy import special as sp
from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA
import matplotlib.pyplot as plt
from itertools import product,starmap
from Utils import Progress, load_eigpairs, mybessel

def UnscreenedCoulombKernel(x,y):
    return np.where((x==0)*(y==0),0,1/np.sqrt(x**2+y**2))

def UnscreenedCoulombKernelFourier(kx,ky):
    return np.where((kx==0)*(ky==0),0,2*np.pi/np.sqrt(kx**2+ky**2))

class CoulombConvolver(numrec.QuickConvolver):
    def __init__(self,xs,ys,kernel_func=UnscreenedCoulombKernel,bc='open'):

        dx=np.mean(np.abs(np.diff(xs)))
        dy=np.mean(np.abs(np.diff(ys)))

        size=(xs.max()-xs.min(),\
              ys.max()-ys.min()) #This is all enough to enable calculating kernel on appropriate origin-centered grid

        pad_mult=np.zeros((3,3))+1
        #Use method of images to induce zero potential at boundaries
        if bc=='closed':
            pad_with='mirror'
            pad_mult[0,1]=-1
            pad_mult[1,0]=pad_mult[1,2]=-1
            pad_mult[2,1]=-1
        elif bc=='open':
            pad_with=0
        else: assert bc in ('open','closed')

        return super().__init__(size=size,kernel_function=kernel_func,\
                                   shape=(len(xs),len(ys)),pad_by=.5,\
                                   pad_with=pad_with,pad_mult=pad_mult)

# Use this one
class CoulombConvolver2(numrec.QuickConvolver):
    def __init__(self,xs,ys,kernel_func_fourier=UnscreenedCoulombKernelFourier,bc='open'):

        size=(xs.max()-xs.min(),\
              ys.max()-ys.min()) #This is all enough to enable calculating kernel on appropriate origin-centered grid

        pad_mult=np.zeros((3,3))+1
        #Use method of images to induce zero potential at boundaries
        if bc=='closed':
            pad_with='mirror'
            pad_mult[0,1]=-1
            pad_mult[1,0]=pad_mult[1,2]=-1
            pad_mult[2,1]=-1
        elif bc=='open':
            pad_with=0
        else: assert bc in ('open','closed')

        return super().__init__(size=size,kernel_function_fourier=kernel_func_fourier,\
                                   shape=(len(xs),len(ys)),pad_by=.5,\
                                   pad_with=pad_with,pad_mult=pad_mult)

# Because somehow `multiprocessing.Pool` can only receive a module-level function??
def apply_CC(psi):
    global CoulConv
    result=CoulConv(psi)
    result-=np.mean(result) #ensure charge neutrality

    return result

def inner_prod(psi,psi_star):
    return np.sum(psi*psi_star)

def dipolefield(x,y,z,direction=[0,1]):
    "`direction` is a vector with `[\rho,z]` components"

    r=np.sqrt(x**2+y**2+z**2)
    rho=np.sqrt(x**2+y**2)
    rhat_rho=rho/r
    rhat_z=z/r

    return (direction[0]*rhat_rho+direction[1]*rhat_z)/r**2

class Translator:
    """
        Allows for the translation of the center point of functions within an xy mesh space.

        Works by generating a bessel function on
        a much larger xy mesh and then translating/truncating to the
        original mesh.  Nothing fancy, but good for performance.
    """

    def __init__(self,
                 xs=np.linspace(-1,1,101),
                 ys=np.linspace(-1,1,101),
                 f = lambda x,y: sp.jv(0,np.sqrt(x**2+y**2))):

        self.f=f

        self.xs,self.ys=xs,ys
        self.Nx,self.Ny=len(xs),len(ys)
        try: self.dx=np.abs(np.diff(xs)[0])
        except IndexError: self.dx=0
        try: self.dy=np.abs(np.diff(ys)[0])
        except IndexError: self.dy=0
        self.xmin=np.min(xs)
        self.ymin=np.min(ys)

        self.bigNx=2*self.Nx+1
        self.bigNy=2*self.Ny+1

        self.bigXs,self.bigYs=np.ogrid[-self.dx*self.Nx:+self.dx*self.Nx:self.bigNx*1j,
                                           -self.dy*self.Ny:+self.dy*self.Ny:self.bigNy*1j]

        self.bigF=self.f(self.bigXs,self.bigYs)
        self.bigF/=np.sqrt(np.sum(self.bigF**2))

    def __call__(self,x0,y0):

        if self.dx: x0bar=(x0-self.xmin)/self.dx
        else: x0bar=0
        if self.dy: y0bar=(y0-self.ymin)/self.dy
        else: y0bar=0

        x0bar=int(x0bar)
        y0bar=int(y0bar)


        result=self.bigF[self.Nx-x0bar:2*self.Nx-x0bar,\
                         self.Ny-y0bar:2*self.Ny-y0bar]

        return AWA(result,axes=[self.xs,self.ys])

class TipResponse:
    """
        Abstraction of a tip which can generate excitations

        Can output the tip eigenbases and the excitation functions
    """

    def __init__(self,xs=np.linspace(-1,1,101), ys=np.linspace(-1,1,101), q=20, N_tip_eigenbasis=5):
        self.q = q
        self.N_tip_eigenbasis = N_tip_eigenbasis

        self.eigenbasis_translators = self._SetEigenbasisTranslators(xs,ys)

    def _SetEigenbasisTranslators(self,xs,ys):
        tip_eb = []
        N = self.N_tip_eigenbasis
        for n in range(N):
            Q=self.q*(n+1)
            exp_prefactor = np.exp(-2*(n+1)/(N+1))
            A = exp_prefactor*Q**2

            D=2 #The larger D goes, the longer range the tip excitation
            func = lambda x,y: np.exp(-Q/D*np.sqrt(x**2+y**2))*mybessel(1,0,Q,x,y)

            tip_eb.append(Translator(xs=xs,ys=ys,f=func))
        return tip_eb

    def __call__(self,x0,y0):
        tip_eb = [t(x0,y0) for t in self.eigenbasis_translators]
        return AWA(tip_eb, axes=[None,tip_eb[0].axes[0],tip_eb[0].axes[1]])

class SampleResponse:
    """
        Generator of sample response based on an input collection of
        eigenpairs (dictionary of eigenvalues + eigenfunctions).

        Can output a whole set of sample response functions from
        an input set of excitation functions.
    """

    def __init__(self,eigpairs,qp,Qfactor=100,N=100,\
                 eigmultiplicity=None,\
                 coulomb_shortcut=False,coulomb_bc='closed',\
                 debug=True):
        # Setting the easy stuff
        self.debug = debug
        self.N=N

        # Setting the various physical quantities
        self._SetEigenbasis(eigpairs,eigmultiplicity)
        self._TuneEigenbasis(qp)
        self._SetAlpha(Qfactor)
        self._SetCoulombKernel(shortcut=coulomb_shortcut,\
                               bc=coulomb_bc)
        self._SetReflectionMatrix(qp)

    def _SetEigenbasis(self,eigpairs,eigmultiplicity=None):

        print('Setting Eigenbasis...')
        start = time.time()
        # For degeneracy
        if not eigmultiplicity: eigmultiplicity={}
        eigvals = sorted(list(eigpairs.keys()))

        #Normalize the eigenfunctions, and apply any multiplicity!
        eigfuncs=[]; eigmult=[]
        for eigval in eigvals:

            #normalize
            eigfunc=np.array(eigpairs[eigval])
            eigfunc=eigfunc/np.sqrt(np.sum(eigfunc**2))

            eigfuncs.append(eigfunc)
            if eigval in eigmultiplicity:
                eigmult.append(eigmultiplicity[eigval])
            else: eigmult.append(1)

        self.xs,self.ys = eigpairs[eigval].axes
        self.eigfuncs = AWA(eigfuncs,\
                            axes=[eigvals,self.xs,self.ys])
        self.eigvals = self.eigfuncs.axes[0] #sorted
        self.eigmult=AWA(eigmult,axes=[eigvals])
        print("\tTime elapsed:{}".format(time.time()-start))

    def _TuneEigenbasis(self,qp):

        if self.debug: print('Tuning Eigenbasis...')
        start = time.time()
        index=np.argmin(np.abs(self.eigvals-np.abs(qp)**2)) #@ASM2019.12.22 - This is to treat `E` not as the squared eigenvalue, but in units of the eigenvalue (`q_omega)
        ind1=np.max([index-self.N//2,0])
        ind2=ind1+self.N
        if ind2>len(self.eigfuncs):
            ind2 = len(self.eigfuncs)+1
            ind1 = ind2-self.N
        if ind1<0: ind1=0
        self.use_eigfuncs=self.eigfuncs[ind1:ind2]
        self.use_eigvals=self.eigvals[ind1:ind2]
        self.use_eigmult=self.eigmult[ind1:ind2]
        self.N=len(self.use_eigfuncs) #Just in case it was less than the originally provided `N`

        self.Us = np.matrix([eigfunc.ravel() for eigfunc in self.use_eigfuncs]).T
        self.Qs2 = np.diag(self.use_eigvals)
        print("\tTime elapsed:{}".format(time.time()-start))

    def _SetAlpha(self,Qfactor=50):
        #We have that `angle(alpha)=-arctan2(qp2,qp1)=-arctan2(1,Q)`
        self.alpha=np.exp(-1j*np.arctan2(1,Qfactor))

    def _SetCoulombKernel(self,shortcut=False,bc='open',multiprocess=False):
        """
            TODO: I hacked this together in some crazy way to force multiprocessing.Pool to work...
                    Needs to be understood and fixed
        """
        start = time.time()
        if shortcut:
            if self.debug: print('Applying Coulomb kernel (with shortcut)...')
            self.use_Veigfuncs=[2*np.pi/np.sqrt(eigval)*eigfunc \
                                for eigval,eigfunc in zip(self.use_eigvals,self.use_eigfuncs)]
            self.V_mn=np.matrix(np.diag(2*np.pi/np.sqrt(self.use_eigvals)))
            self.V_mn[np.isnan(self.V_mn)]=0
            #no need to ravel, just appropriate `self.Phis`
            self.Ws=self.Us @ self.V_mn

        if not shortcut:
            eigfuncs = self.use_eigfuncs

            global CoulConv
            CoulConv=CoulombConvolver2(self.xs,self.ys,bc=bc)

            if self.debug: print('Applying Coulomb kernel (without shortcut)...')
            #Need a with statement, otherwise an abort fails to close processes and seems to crash the interpreter
            if multiprocess:
                with mp.Pool(8) as self.Pool:
                    self.use_Veigfuncs=list(self.Pool.map(apply_CC,eigfuncs))
                    self.Ws=np.matrix([Veigfunc.ravel() for Veigfunc in self.use_Veigfuncs]).T
                    self.V_mn=list(self.Pool.starmap(inner_prod,\
                                                     product(list(self.use_eigfuncs),list(self.use_Veigfuncs))))
            else:
                self.use_Veigfuncs=list(map(apply_CC,eigfuncs))
                self.Ws=np.matrix([Veigfunc.ravel() for Veigfunc in self.use_Veigfuncs]).T
                self.V_mn=list(starmap(inner_prod,product(list(self.use_eigfuncs),list(self.use_Veigfuncs))))

            self.V_mn=np.matrix(np.array(self.V_mn).reshape((self.N,)*2))
            self.V_mn=(self.V_mn+self.V_mn.T)/2
        print("\tTime elapsed:{}".format(time.time()-start))

    def _SetReflectionMatrix(self,qp,diag=True):
        if self.debug: print('Computing Reflection Matrix...')
        start = time.time()
        self.qp=qp

        if diag: V=np.diag(np.diag(self.V_mn))
        else: V=self.V_mn

        #@ASM2020.04.03: Minus comes from need to define reflection coefficient for Ez field
        num=-self.alpha/(2*np.pi)*(self.Qs2)
        inv=np.linalg.inv(self.qp*np.identity(self.Qs2.shape[0]) \
                          -self.alpha/(2*np.pi)*(V @ self.Qs2))

        self.R_mn = inv @ num

        #Weight matrix elements by multiplicity of the eigenfunction
        Mult=np.diag(self.use_eigmult)
        self.R_mn = self.R_mn @ Mult
        print("\tTime elapsed:{}".format(time.time()-start))

        #self.R_rmrn = self.Ws @ self.R_mn @ self.Us.T

    def GetReflectionCoefficient(self,tip_eigenbasis):

        self.Psis=np.matrix([eigfunc.ravel() for eigfunc in tip_eigenbasis]).T #Only have to unravel sample eigenfunctions once (twice speed-up)
        self.PsisInv=np.linalg.pinv(self.Psis)

        self.P_nk = self.Us.T @ self.Psis
        self.P_jm = self.PsisInv @ self.Ws

        self.R_jk = self.P_jm @ self.R_mn @  self.P_nk
        #self.R_jk = self.PsisInv @ self.R_rmrn @ self.Psis

        return self.R_jk

    def __call__(self,excitations):#,U,tip_eigenbasis):
        """This will evaluate the total potential"""

        if np.array(excitations).ndim==2: excitations=[excitations]
        Exc=np.matrix([exc.ravel() for exc in excitations]).T
        self.P_nk = self.Us.T @ Exc
        result=self.Ws @ self.R_mn @ self.P_nk

        result=np.array(result).T.reshape((len(excitations),\
                                           len(self.xs),len(self.ys)))

        return AWA(result,axes=[None,self.xs,self.ys],\
                   axis_names=['Excitation index','x','y']).squeeze() #, AWA(projected_result,axes=[None,self.xs,self.ys]).squeeze()

    def getXYGrids(self):

        Xs=self.xs
        Ys=self.ys

        Xs=Xs.reshape((len(Xs),1))
        Ys=Ys.reshape((1,len(Ys)))

        return Xs,Ys
