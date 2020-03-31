import os,h5py,time
import Plasmon_Modeling as PM
import multiprocessing as mp
import numpy as np
from scipy import special as sp
from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA
import matplotlib.pyplot as plt
from itertools import product
from Utils import Progress, load_eigpairs

def Calc(psi,psi_star):
    return np.sum(psi_star*myQC(psi))

def mybessel(A,v,Q,x,y):
    r = np.sqrt(x**2+y**2)
    return A*sp.jv(v,Q*r)

def planewave(qx,qy,x,y):
    return np.sin(qx*x+qy*y)

class Translator:
    """
        Allows for the translation of the center point of functions within an xy mesh space.

        Works by generating a bessel function on
        a much larger xy mesh and then translating/truncating to the
        original mesh.  Nothing fancy, but good for performance.
    """

    """
        TODO: Something is going on that flips the x and y coordinates when plotting or translating.

        Doesn't make much sense...
    """

    def __init__(self,
                 xs=np.linspace(-1,1,101),
                 ys=np.linspace(-1,1,101),
                 f = lambda x,y: sp.jv(0,np.sqrt(x**2+y**2))):

        self.f = f

        #Bookkeeping of the coordinate mesh
        self.xs,self.ys=xs,ys
        self.shape=(len(xs),len(ys))
        self.midx=self.xs[self.shape[0]//2]
        self.midy=self.ys[self.shape[1]//2]
        self.dx=np.max(self.xs)-np.min(self.xs)
        self.dy=np.max(self.ys)-np.min(self.ys)

        #Make a mesh grid twice bigger in each direction
        bigshape=[2*N-1 for N in self.shape]
        xs2grid,ys2grid=np.ogrid[-self.dx:+self.dx:bigshape[0]*1j,
                                 -self.dy:+self.dy:bigshape[1]*1j]

        self.bigF = self.f(xs2grid,ys2grid)

    def __call__(self,x0,y0):
        shift_by_dx=x0-self.midx
        shift_by_dy=y0-self.midy
        shift_by_nx=int(self.shape[0]*shift_by_dx/self.dx)
        shift_by_ny=int(self.shape[1]*shift_by_dy/self.dy)
        newJ=np.roll(np.roll(self.bigF,shift_by_nx,axis=0),\
                     shift_by_ny,axis=1)
        output = newJ[self.shape[0]//2:(3*self.shape[0])//2,\
                     self.shape[1]//2:(3*self.shape[1])//2]
        return AWA(output,axes=[self.xs,self.ys])

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
            Q = (2*self.q*(n+1)/(N+1))
            exp_prefactor = np.exp(-2*(n+1)/(N+1))
            A = exp_prefactor*Q**2

            func = lambda x,y: np.exp(np.sqrt(x**2+y**2))*mybessel(A,0,Q,x,y)

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

    def __init__(self,eigpairs,E,N=100,debug=True):
        # Setting the easy stuff
        eigvals = list(eigpairs.keys())
        eigfuncs = list(eigpairs.values())
        self.debug = debug
        self.xs,self.ys = eigfuncs[0].axes
        self.eigfuncs = AWA(eigfuncs,\
                          axes=[eigvals,self.xs,self.ys]).sort_by_axes()
        self.eigvals = self.eigfuncs.axes[0]
        self.phishape = self.eigfuncs[0].shape
        self.E = E
        self.N = N

        # Setting the various physical quantities
        self._SetUseEigenvalues(E)
        self._SetEnergy()
        self._SetSigma(10,10)
        self._SetCoulombKernel()
        self._SetScatteringMatrix()

    def _SetEnergy(self):
        """
            TODO: check energy units and sqrt eigenvals
        """
        if self.debug: print('Setting Energy')
        self.Phis=np.matrix([eigfunc.ravel() for eigfunc in self.use_eigfuncs])
        self.Q = np.diag(self.use_eigvals)

    def _SetUseEigenvalues(self, E):
        if self.debug: print('Setting Use Eigenvalues')
        index=np.argmin(np.abs(np.sqrt(self.eigvals)-E)) #@ASM2019.12.22 - This is to treat `E` not as the squared eigenvalue, but in units of the eigenvalue (`q_omega)
        ind1=np.max([index-self.N//2,0])
        ind2=ind1+self.N
        if ind2>len(self.eigfuncs):
            ind2 = len(self.eigfuncs)
            ind1 = ind2-self.N
        self.use_eigfuncs=self.eigfuncs[ind1:ind2]
        self.use_eigvals=self.eigvals[ind1:ind2]

    def _SetSigma(self,L,lamb):
        if self.debug: print('Setting Sigma')

        self.sigma = PM.S()
        self.sigma.set_sigma_values(lamb, L)
        sigma_tilde = self.sigma.get_sigma_values()[0]+1j*self.sigma.get_sigma_values()[1]
        self.alpha = -1j*sigma_tilde/np.abs(sigma_tilde)

        """
            TODO: Do we need to change this? As of 2019.12.21, this was a placeholder
        """
        self.alpha = 1 #@ASM2019.12.21 just for nwo we put the 'complexity' into input `E`, until we get serious about recasting it to `q_omega`

    def _SetCoulombKernel(self):
        """
            TODO: I hacked this together in some crazy way to force multiprocessing.Pool to work...
                    Needs to be understood and fixed
        """
        if self.debug: print('Setting Kernel')
        poorman = False
        self.V_nm = np.zeros([len(self.use_eigvals), len(self.use_eigvals)])
        if not poorman:
            eigfuncs = self.use_eigfuncs
            kern_func = lambda x,y: 1/np.sqrt(x**2+y**2+1e-4)
            global myQC
            size=(self.xs.max()-self.xs.min(),self.ys.max()-self.ys.min())
            myQC=numrec.QuickConvolver(size=size,kernel_function=kern_func,\
                                       shape=eigfuncs[0].shape,pad_by=.5,pad_with=0)
            p = mp.Pool(8)
            self.V_nm = np.array(p.starmap(Calc,product(eigfuncs,eigfuncs))).reshape((self.N,self.N))
        else:
            for i,v in enumerate(self.use_eigvals):
                self.V_nm[i,i] = 1/np.sqrt(v)

    def _SetScatteringMatrix(self):
        if self.debug: print('Setting Scattering Matrix')
        self.D = self.E*np.linalg.inv(self.E*np.identity(self.Q.shape[0]) - self.alpha*self.Q.dot(self.V_nm))

    def GetRAlphaBeta(self,tip_eigenbasis):
        Psi=np.matrix([eigfunc.ravel() for eigfunc in tip_eigenbasis]) #Only have to unravel sample eigenfunctions once (twice speed-up)
        U=Psi*self.Phis.T
        U_inv = np.linalg.pinv(U) #@ASM2019.12.21 good to use inverse, since tip-basis may not be orthogonal
        result=np.dot(U,np.dot(self.D,U_inv))

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

def TestScatteringBasisChange(q=44,\
                           E=44*np.exp(1j*2*np.pi*5e-2),\
                           N_sample_eigenbasis=100,\
                           N_tip_eigenbasis = 10):

    global Responder,Tip,R_alphabeta

    Responder = SampleResponse(eigpairs,E=E,N=N_sample_eigenbasis)
    xs,ys = Responder.xs,Responder.ys
    Tip = TipResponse(xs,ys,q=q,N_tip_eigenbasis=N_tip_eigenbasis)

    betaz_alpha = np.diag((2-.1j)*(np.arange(N_tip_eigenbasis)+1))
    Lambdaz_beta = ((1+np.arange(N_tip_eigenbasis))[::-1])

    Ps=np.zeros((len(xs),len(ys)))
    Rs=np.zeros((len(xs),len(ys)))
    last = 0
    for i,x0 in enumerate(xs):
        for j,y0 in enumerate(ys):
            start = time.time()

            tip_eigenbasis = Tip(x0,y0)
            R_alphabeta = Responder.GetRAlphaBeta(tip_eigenbasis)
            Ps[i,j] = np.sum(np.linalg.inv(betaz_alpha-R_alphabeta).dot(Lambdaz_beta))
            Rs[i,j] = np.sum(np.diag(R_alphabeta))/N_tip_eigenbasis

            last = Progress(i,len(xs),last)

            #if i==0 and j==0:
            #    plt.figure()
            #    plt.imshow(np.abs(R_alphabeta))
            #    plt.show()

    return {'P':Ps,'R':Rs}

show_eigs = False
run_test = True
global eigpairs
xs,ys = np.linspace(-1,1,101), np.linspace(-1,1,101)
xv,yv = np.meshgrid(xs,ys)
#eigpairs = load_eigpairs(basedir="../sample_eigenbasis_data")
eigpairs = {}
for n in range(1,300):
    qx = np.pi*n
    eigpairs[qx] = AWA(planewave(qx,0,xv,yv), axes = [xs,ys])
if show_eigs:
    for i,q in enumerate(list(eigpairs.keys())):
        if i<5:
            plt.figure()
            plt.imshow(eigpairs[q])
            plt.title("q={}".format(q))
    plt.show()

if run_test:
    d=TestScatteringBasisChange(q=44,N_tip_eigenbasis=2)
    plt.figure()
    plt.imshow(np.abs(d['P'])); plt.title('P');plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(d['R'])); plt.title('R');plt.colorbar()
    plt.show()
