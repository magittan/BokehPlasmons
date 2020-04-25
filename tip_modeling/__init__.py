import warnings
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import time

from scipy import special as sp
from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA
from itertools import product,starmap
from BokehPlasmons.tip_modeling.utils import *

warnings.filterwarnings('ignore')

DEBUG = True

# Because somehow `multiprocessing.Pool` can only receive a module-level function??
def apply_CC(psi):
    result=CoulConv(psi)
    result-=np.mean(result) #ensure charge neutrality

    return result

class CoulombConvolver(numrec.QuickConvolver):
    def __init__(self,xs,ys,kernel_func_fourier=unscreened_coulomb_kernel_fourier,bc='periodic'):

        size=(xs.max()-xs.min(),\
              ys.max()-ys.min()) #This is all enough to enable calculating kernel on appropriate origin-centered grid

        pad_mult=np.zeros((3,3))+1
        #Use method of images to induce zero potential at boundaries
        if bc=='closed':
            pad_with='mirror'
            pad_mult[0,1]=-1
            pad_mult[1,0]=pad_mult[1,2]=-1
            pad_mult[2,1]=-1
            pad_by=0.5
        elif bc=='open':
            pad_with=0; pad_by=0.5
        elif bc=='periodic':
            pad_with=0; pad_by=0
        else: assert bc in ('open','closed','periodic')

        return super().__init__(size=size,kernel_function_fourier=kernel_func_fourier,\
                                   shape=(len(xs),len(ys)),pad_by=pad_by,\
                                   pad_with=pad_with,pad_mult=pad_mult)

class Translator:
    """
        Allows for the translation of the center point of functions within an xy mesh space.

        Works by evaluating a function at center of a much larger
        auxiliary xy mesh and then translating/truncating to the
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
    
class TranslatorPeriodic:
    """
        Allows for the translation of the center point of functions within an xy mesh space.

        Mesh-periodic version of `Translator`.
        
        Works by evaluating a function at center of an
        auxiliary xy mesh and then translating/truncating to the
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
        
        if self.Nx % 2: #if odd
            self.Ndxleft=self.Ndxright=(self.Nx-1)/2
        else: #Zero coordinate will start second half
            self.Ndxleft=self.Nx/2; self.Ndxright=self.Ndxleft-1
            
        if self.Ny % 2: #if odd
            self.Ndyleft=self.Ndyright=(self.Ny-1)/2
        else: #Zero coordinate will start second half
            self.Ndyleft=self.Ny/2; self.Ndyright=self.Ndyleft-1

        #zero coordinate is now guaranteed in this "larger" mesh
        self.bigXs,self.bigYs=np.ogrid[-self.dx*self.Ndxleft:+self.dx*self.Ndxright:self.Nx*1j,
                                       -self.dy*self.Ndyleft:+self.dy*self.Ndyright:self.Ny*1j]

        self.bigF=self.f(self.bigXs,self.bigYs)
        self.bigF/=np.sqrt(np.sum(self.bigF**2))

    def __call__(self,x0,y0):

        #`x0bar,y0bar` this is the index where we want the zero coordinate
        if self.dx: x0bar=(x0-self.xmin)/self.dx
        else: x0bar=0
        if self.dy: y0bar=(y0-self.ymin)/self.dy
        else: y0bar=0

        Nxshift=int(x0bar-self.Ndxleft)
        Nyshift=int(y0bar-self.Ndyleft)
        result=np.roll(self.bigF,Nxshift,axis=0)
        result=np.roll(result,Nyshift,axis=1)

        return AWA(result,axes=[self.xs,self.ys])

class TipResponse:
    """
        Abstraction of a tip which can generate excitations

        Can output the tip eigenbases and the excitation functions
    """

    def __init__(self,
                xs=np.linspace(-1,1,101), ys=np.linspace(-1,1,101),
                q=20, N_tip_eigenbasis=5,
                func = faketip,
                periodic=False):
        self.q = q
        self.N_tip_eigenbasis = N_tip_eigenbasis
        self.func = func
        self.eigenbasis_translators = self._set_eigenbasis_translators(xs,ys,\
                                                                       periodic=periodic)

    def _set_eigenbasis_translators(self,xs,ys,periodic=False):
        tip_eb = []
        N = self.N_tip_eigenbasis
        if periodic: use_Translator=TranslatorPeriodic
        else: use_Translator=Translator
        
        for n in range(N):
            tip_eb.append(use_Translator(xs=xs,ys=ys,f=self.func(self.q,n,N,xs,ys)))
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

    def __init__(self,eigpairs,qp,
                    Qfactor=100,N=100,
                    eigmultiplicity=None,
                    coulomb_shortcut=False,coulomb_bc='closed',
                    coulomb_multiprocess=False,
                    debug=True):
        # Setting the easy stuff
        self.debug = debug
        self.N=N

        # Setting the various physical quantities
        self._set_eigenbasis(eigpairs,eigmultiplicity)
        self._tune_eigenbasis(qp)
        self._set_alpha(Qfactor)
        self._set_coulomb_kernel(shortcut=coulomb_shortcut,\
                                 bc=coulomb_bc,\
                                 multiprocess=coulomb_multiprocess)
        self._set_reflection_matrix(qp)

    def _set_eigenbasis(self,eigpairs,eigmultiplicity=None):

        print('Setting Eigenbasis...')
        T=Timer()
        # For degeneracy
        if not eigmultiplicity: eigmultiplicity={}
        eigvals = sorted(list(eigpairs.keys()))

        #Normalize the eigenfunctions, and apply any multiplicity!
        eigfuncs=[]; eigmult=[]
        for eigval in eigvals:

            #normalize
            eigfunc=np.array(eigpairs[eigval])
            eigfunc=normalize(eigfunc)

            eigfuncs.append(eigfunc)
            if eigval in eigmultiplicity:
                eigmult.append(eigmultiplicity[eigval])
            else: eigmult.append(1)

        self.xs,self.ys = eigpairs[eigval].axes
        self.eigfuncs = AWA(eigfuncs,\
                            axes=[eigvals,self.xs,self.ys])
        self.eigvals = self.eigfuncs.axes[0] #sorted
        self.eigmult=AWA(eigmult,axes=[eigvals])
        T()

    def _tune_eigenbasis(self,qp):

        if self.debug: print('Tuning Eigenbasis...')
        T=Timer()
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
        T()

    def _set_alpha(self,Qfactor=50):
        #We have that `angle(alpha)=-arctan2(qp2,qp1)=-arctan2(1,Q)`
        self.alpha=np.exp(-1j*np.arctan2(1,Qfactor))

    def _set_coulomb_kernel(self,shortcut=False,bc='open',multiprocess=False):
        T=Timer()
        if shortcut:
            if self.debug: print('Applying Coulomb kernel (with shortcut)...')
            self.use_Veigfuncs=[2*np.pi/np.sqrt(eigval)*eigfunc \
                                for eigval,eigfunc in zip(self.use_eigvals,self.use_eigfuncs)]
            self.V_mn=np.matrix(np.diag(2*np.pi/np.sqrt(self.use_eigvals)))
            self.V_mn[np.isnan(self.V_mn)]=0
            #no need to ravel, just appropriate `self.Us`
            self.Ws=(self.V_mn @ self.Us.T).T
        
        if not shortcut:
            eigfuncs = [np.array(eigfunc) for eigfunc in self.use_eigfuncs]
            
            global CoulConv
            CoulConv=CoulombConvolver(self.xs,self.ys,bc=bc)
                
            if self.debug: print('Applying Coulomb kernel...')
            #Need a with statement, otherwise an abort fails to close processes and seems to crash the interpreter
            # this guy's reliability is nightmarish.. how about joblib?
            # https://joblib.readthedocs.io/en/latest/
            #Also, IO seems to stall when the eigenfunctions are large.  Howabout shared memory instead?
            # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html 
            if multiprocess: 
                with mp.Pool(None) as Pool:
                    self.use_Veigfuncs=list(Pool.map(apply_CC,eigfuncs))
                    Veigfuncs=[np.array(Veigfunc) for Veigfunc in self.use_Veigfuncs] #demote to temporary arrays for speed
                    self.Ws=np.matrix([Veigfunc.ravel() for Veigfunc in Veigfuncs]).T
                    self.V_mn=list(Pool.starmap(inner_prod,\
                                                product(eigfuncs,Veigfuncs)))
                    self.V_mn=np.matrix(np.array(self.V_mn).reshape((self.N,)*2))
            else:
                self.use_Veigfuncs=list(map(apply_CC,eigfuncs))
                Veigfuncs=[np.array(Veigfunc) for Veigfunc in self.use_Veigfuncs] #demote to temporary arrays for speed
                
                self.Ws=np.matrix([Veigfunc.ravel() for Veigfunc in Veigfuncs]).T
                self.V_mn=build_matrix(eigfuncs,Veigfuncs)
                #self.V_mn=list(starmap(inner_prod,product(eigfuncs,Veigfuncs)))
                #self.V_mn=np.matrix(np.array(self.V_mn).reshape((self.N,)*2))
            
            self.V_mn=(self.V_mn+self.V_mn.T)/2
        T()

    def _set_reflection_matrix(self,qp,diag=True):
        if self.debug: print('Computing Reflection Matrix...')
        T=Timer()
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
        T()

    def get_reflection_coefficient(self,tip_eigenbasis):
        
        #Prepare the array of input excitations
        tip_eigenbasis=np.array(tip_eigenbasis)
        if tip_eigenbasis.ndim==2:
            tip_eigenbasis=tip_eigenbasis.reshape((1,)+tip_eigenbasis.shape)

        self.Psis=np.matrix([eigfunc.ravel() for eigfunc in tip_eigenbasis]).T #Only have to unravel sample eigenfunctions once (twice speed-up)
        self.PsisInv=np.linalg.pinv(self.Psis)

        self.P_nk = self.Us.T @ self.Psis
        self.P_jm = self.PsisInv @ self.Ws

        self.R_jk = self.P_jm @ self.R_mn @  self.P_nk
        #self.R_jk = self.PsisInv @ self.R_rmrn @ self.Psis

        return self.R_jk

    def get_xy_grids(self):
        Xs=self.xs
        Ys=self.ys

        Xs=Xs.reshape((len(Xs),1))
        Ys=Ys.reshape((1,len(Ys)))

        return Xs,Ys

    def raster_scan(self, TipEigenbasis,\
                    xs=None,ys=None,beta=0):
        N_tip_eigenbasis = TipEigenbasis.N_tip_eigenbasis
        tip_eigenpoles = np.diag((2-.1j)*np.exp(np.arange(N_tip_eigenbasis)))
        tip_eigenresidues = 1+np.arange(N_tip_eigenbasis) #np.exp(np.arange(N_tip_eigenbasis))
        
        if xs is None: xs=self.xs
        elif not hasattr(xs,'__len__'): xs=[xs]
        if ys is None: ys=self.ys
        elif not hasattr(ys,'__len__'): ys=[ys]

        Ps=np.zeros((len(xs),len(ys)),dtype=np.complex)
        Rs=np.zeros((len(xs),len(ys)),dtype=np.complex)
        
        #Beta remains untested
        I=np.eye(N_tip_eigenbasis)
        Beta=I*beta
    
        last = 0
        print("Raster scanning tip...")
        T=Timer()
        for i,x0 in enumerate(xs):
            for j,y0 in enumerate(ys):
                tip_eigenbasis = TipEigenbasis(x0,y0)
                self.tip_eigenbasis=tip_eigenbasis
                R2D = self.get_reflection_coefficient(tip_eigenbasis)
            
                Rsample = Beta + R2D @ (I-Beta)

                P_sample=np.sum(np.linalg.inv(tip_eigenpoles-Rsample).dot(tip_eigenresidues))
                P_0=np.sum(np.linalg.inv(tip_eigenpoles).dot(tip_eigenresidues))

                Ps[i,j] = P_sample-P_0
                Rs[i,j] = np.sum(np.diag(Rsample))/N_tip_eigenbasis

                last = progress(i,len(xs),last)

        T()
        return {'P':AWA(Ps,axes=[xs,ys],axis_names=['x','y']).squeeze(),\
                'R':AWA(Rs,axes=[xs,ys],axis_names=['x','y']).squeeze()}

    def __call__(self,excitations):#,U,tip_eigenbasis):
        """This will evaluate the induced potential"""

        if np.array(excitations).ndim==2: excitations=[excitations]
        Exc=np.matrix([exc.ravel() for exc in excitations]).T
        self.P_nk = self.Us.T @ Exc
        result=self.Ws @ self.R_mn @ self.P_nk

        result=np.array(result).T.reshape((len(excitations),\
                                           len(self.xs),len(self.ys)))

        return AWA(result,axes=[None,self.xs,self.ys],\
                   axis_names=['Excitation index','x','y']).squeeze() #, AWA(projected_result,axes=[None,self.xs,self.ys]).squeeze()
        
    def get_coulomb_eigenbasis(self):
        """It was thought that maybe generated eigenfunctions were not sufficiently orthonormal
        for large matrices to avoid off-diagonal elements.  But it's now seen that this function
        does not affect results."""
        
        T=Timer()
        if self.debug: print('Diagonalizing composite operator...')
        
        eigfunc_shape=self.use_eigfuncs[0].shape
        
        self.VL_mn= self.V_mn @ self.Qs2 #These roughly commute anyway
        
        eigvals,eigvecs=np.linalg.eigh(self.VL_mn) #Use `eigh` so that the eigenvalues are sorted
        self.VL_eigvals=(eigvals.real)
        
        #We project onto Us for now because we don't know what we're diagonalizing
        self.VL_Us = np.matrix(self.Us @ eigvecs) #Hit `self.Us` with all our column vectors
        self.VL_eigfuncs=[AWA(u.reshape(eigfunc_shape),\
                               axes=[self.xs,self.ys]) \
                           for u in self.VL_Us.T]
        
        T()
            
        return self.VL_eigfuncs
    
        
    def perfectly_reflect(self,excitations):
        """This is just like `__call__` except without any sample reflection matrix."""
        
        if np.array(excitations).ndim==2: excitations=[excitations]
        Exc=np.matrix([exc.ravel() for exc in excitations]).T
        
        self.P_nk = self.Us.T @ Exc
        result=self.Us @ self.P_nk
        
        result=np.array(result).T.reshape((len(excitations),\
                                           len(self.xs),len(self.ys)))
            
        return AWA(result,axes=[None,self.xs,self.ys],\
                   axis_names=['Excitation index','x','y']).squeeze()
