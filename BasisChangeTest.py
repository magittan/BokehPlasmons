import h5py,time, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.integrate import nquad
from numba import jit, cuda, prange

def ChangeBasis(to_eigenbasis,from_eigenbasis,xs,ys):
    to_nn = range(to_eigenbasis.shape[0])
    from_nn = range(from_eigenbasis.shape[0])
    U = np.zeros((len(to_nn), len(from_nn)))
    for to_n in to_nn:
        for from_n in from_nn:
            from_ef = to_eigenbasis[to_n]
            to_ef = from_eigenbasis[from_n]
            dot_prod = np.asarray(from_ef*to_ef)
            sum = np.trapz(np.trapz(dot_prod,ys),xs)
            #sum = nquad(lambda x,y: IntFun(dot_prod,x,y),[[-1,1],[-1,1]])
            U[to_n,from_n] = sum
    return U

@jit
def ChangeBasisCPU(to_eigenbasis,from_eigenbasis,xs,ys):
    to_nn = range(to_eigenbasis.shape[0])
    from_nn = range(from_eigenbasis.shape[0])
    U = np.zeros((len(to_nn), len(from_nn)))
    for to_n in to_nn:
        for from_n in from_nn:
            from_ef = to_eigenbasis[to_n]
            to_ef = from_eigenbasis[from_n]
            dot_prod = np.asarray(from_ef*to_ef)
            sum = np.trapz(np.trapz(dot_prod,ys),xs)
            #sum = nquad(lambda x,y: IntFun(dot_prod,x,y),[[-1,1],[-1,1]])
            U[to_n,from_n] = sum
    return U

@cuda.jit
def ChangeBasisGPU(U,dot_prod,to_eigenbasis,from_eigenbasis,xs,ys):
    to_nn = range(to_eigenbasis.shape[0])
    from_nn = range(from_eigenbasis.shape[0])
    sx, sy = to_eigenbasis[0].shape
    for to_n in to_nn:
        for from_n in from_nn:
            from_ef = to_eigenbasis[to_n]
            to_ef = from_eigenbasis[from_n]
            #dot_prod = np.reshape(np.dot(from_ef,to_ef.T),(sx,sy))
            for i in range(dot_prod.shape[0]):
                for j in range(dot_prod.shape[1]):
                    dot_prod[i][j] = from_ef[i,j]*to_ef[i,j]
            sum = np.trapz(np.trapz(dot_prod,ys),xs)
            #sum = nquad(lambda x,y: IntFun(dot_prod,x,y),[[-1,1],[-1,1]])
            U[to_n,from_n] = sum
            for i in range(dot_prod.shape[0]):
                for j in range(dot_prod.shape[1]):
                    dot_prod[i,j] = 0
    #return U

def ImshowAll(eigenbasis):
    N = eigenbasis.shape[0]
    ncols = 4
    nrows = int(np.ceil(N/ncols))
    fig,ax = plt.subplots(nrows,ncols)
    for i in range(nrows):
        for j in range(ncols):
            idx = ncols*i+j
            if idx < N:
                ax[i,j].imshow(eigenbasis[idx],origin='lower',extent=[-1,1,-1,1])
    plt.show()

def VInBasis(vec,basis):
    basis_shape = basis[0].shape
    sum = np.zeros(basis_shape)
    for i,vi in enumerate(vec):
        sum += vi*basis[i]
    return sum

def IntFun(fun,x,y):
    x = int((x+1)*50)
    y = int((y+1)*50)
    return fun[x,y]

def LoadEigenbasis(fname, dirname = './'):
    full_fname = os.path.join(dirname, fname)
    sample_eigenpairs = {}
    with h5py.File(full_fname,'r') as f:
        for key in list(f.keys()):
            sample_eigenpairs[float(key)] = np.array(f.get(key))
    return sample_eigenpairs

def main():
    sample_eigenpairs = LoadEigenbasis('UnitSquareMesh_100x100_1000_eigenbasis.h5')
    sample_eigenvalues = sorted(list(sample_eigenpairs.keys()))

    extent = [-1,1,-1,1]
    nx, ny=101,101
    xs = np.linspace(-1,1,nx)
    ys = np.linspace(-1,1,ny)
    xv,yv = np.meshgrid(xs,ys,sparse=True)
    rs = np.sqrt(xv**2+yv**2)
    theta = np.arctan(yv/xv)
    theta[np.where(np.isnan(theta))] = 0
    times = {}

    #for n in range(200):
    for n in [100]:
        N_tip_eigenfunctions = n # Dimension of tip eigenbasis; in this case we use bessel functions
        N_sample_eigenfunctions = len(sample_eigenvalues)

        # Construct tip and sample eigenbases
        tip_eigenbasis, sample_eigenbasis = [], []
        for n in range(N_sample_eigenfunctions):
            sample_eigenbasis.append(sample_eigenpairs[sample_eigenvalues[n]])

        for alpha in range(N_tip_eigenfunctions):
            tip_eigenbasis.append(jv(alpha//4,10*rs)*np.cos((alpha%4)*theta))

        sample_eigenbasis = np.asarray(sample_eigenbasis)
        tip_eigenbasis = np.asarray(tip_eigenbasis)
        #ImshowAll(tip_eigenbasis)

        # Construct change of basis matrix U
        start = time.time()
        U_tip_to_sample = ChangeBasis(sample_eigenbasis,tip_eigenbasis,xs,ys)
        elapsed = time.time()-start
        print("Basis Change (tip to sample): {} seconds".format(elapsed))
        times[N_tip_eigenfunctions*N_sample_eigenfunctions] = elapsed
    """
    x,y = [],[]
    for k in times.keys():
        x.append(k)
        y.append(times[k])
    plt.figure()
    plt.plot(x,y)
    plt.show()
    """

    tip_v = np.ones((N_tip_eigenfunctions))
    tip_m = VInBasis(tip_v,tip_eigenbasis)
    sample_v = U_tip_to_sample.dot(tip_v.T)
    sample_m = VInBasis(sample_v,sample_eigenbasis)

    max_ratio = np.max(tip_m)/np.max(sample_m)
    diff = np.sum((sample_m-tip_m)**2)/np.prod(sample_m.shape)
    print('Difference: {}'.format(diff))

    plt.figure()
    plt.subplot(121)
    plt.imshow(sample_m*max_ratio,origin='lower',extent=extent)
    plt.title('Sample Basis')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(tip_m,origin='lower', extent = extent)
    plt.title('Tip Basis')
    plt.colorbar()
    plt.show()
    """
    """

if __name__=='__main__': main()
