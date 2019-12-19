import h5py,time,os, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.integrate import nquad
from numba import jit, cuda, prange

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

def ChangeBasisGPU(to_eigenbasis,from_eigenbasis,xs,ys):
    TPB = 16
    threadsperblock = (TPB,TPB)

    N_to_eigfunc = len(to_eigenbasis)
    N_from_eigfunc = len(from_eigenbasis)
    blockspergrid_x = int(math.ceil(N_to_eigfunc/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(N_from_eigfunc/threadsperblock[0]))
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    d_to_eigenbasis = cuda.to_device(np.ascontiguousarray(to_eigenbasis))
    d_from_eigenbasis = cuda.to_device(np.ascontiguousarray(from_eigenbasis))
    d_U = cuda.device_array((N_to_eigfunc,N_from_eigfunc))
    _ChangeBasisGPU[blockspergrid,threadsperblock](d_U,d_to_eigenbasis,d_from_eigenbasis,xs,ys)
    return d_U.copy_to_host()

@cuda.jit
def _ChangeBasisGPU(U, to_eigenbasis,from_eigenbasis,xs,ys):
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

def TestBasisChange(U,from_eigenbasis,to_eigenbasis):
    from_v = OneHot(0,U.shape[1])+OneHot(1,U.shape[1])+OneHot(2,U.shape[1])
    to_v = U.dot(from_v.T)

    from_m = VInBasis(from_v,from_eigenbasis)
    to_m = VInBasis(to_v,to_eigenbasis)

    ImshowSampleAndTip(to_m,from_m)

# Imshow functions
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

def ImshowSampleAndTip(sample_m,tip_m,origin='lower',extent=None):
    plt.figure()
    plt.subplot(121)
    plt.imshow(sample_m,origin=origin,extent=extent)
    plt.title('Sample Basis')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(tip_m,origin=origin, extent = extent)
    plt.title('Tip Basis')
    plt.colorbar()
    plt.show()

# Eigenbasis functions
def ConstructEigenbases(N_tip_efs,N_sample_efs,rs,theta,sample_evs,sample_eps):
    tip_eigenbasis, sample_eigenbasis = [],[]

    for n in range(N_sample_efs):
        sample_eigenbasis.append(sample_eps[sample_evs[n]])

    for alpha in range(N_tip_efs):
        tip_eigenbasis.append(jv(alpha//10,10*rs)*np.cos((alpha%10)*theta))

    sample_eigenbasis = np.asarray(sample_eigenbasis)
    tip_eigenbasis = np.asarray(tip_eigenbasis)
    return tip_eigenbasis,sample_eigenbasis

def LoadEigenbasis(fname, dirname = './'):
    full_fname = os.path.join(dirname, fname)
    eigenpairs = {}
    with h5py.File(full_fname,'r') as f:
        for key in list(f.keys()):
            eigenpairs[float(key)] = np.array(f.get(key))
    return eigenpairs

def VInBasis(vec,basis):
    basis_shape = basis[0].shape
    sum = np.zeros(basis_shape)
    for i,vi in enumerate(vec):
        sum += vi*basis[i]
    return sum

# Miscellaneous functions
def IntFun(fun,x,y):
    x = int((x+1)*50)
    y = int((y+1)*50)
    return fun[x,y]

def OneHot(i,L):
    onehot = np.concatenate((np.array([1]),np.zeros(L-1)))
    onehot = np.roll(onehot,i)
    return onehot

def Progress(i,L,last):
    next = last+10
    percent = 100*i/L
    if percent >= next:
        print('{}% complete...'.format(next))
        return next
    else:
        return last

def main():
    sample_eigenpairs = LoadEigenbasis('UnitSquareMesh_100x100_1000_eigenbasis.h5')
    sample_eigenvalues = sorted(list(sample_eigenpairs.keys()))
    TPB = 16

    extent = [-1,1,-1,1]

    nx, ny = 101,101
    xs, ys = np.linspace(-1,1,nx), np.linspace(-1,1,ny)
    xv, yv = np.meshgrid(xs,ys,sparse=True)

    rs = np.sqrt(xv**2+yv**2)
    theta = np.arctan(yv/xv)
    theta[np.where(np.isnan(theta))] = 0

    # Construct tip and sample eigenbases
    N_tip_eigenfunctions = 100
    N_sample_eigenfunctions = 1000
    tip_eigenbasis, sample_eigenbasis = ConstructEigenbases(N_tip_eigenfunctions,N_sample_eigenfunctions,rs,theta,sample_eigenvalues,sample_eigenpairs)

    # Construct change of basis matrix U
    start = time.time()
    U_tip_to_sample = ChangeBasis(sample_eigenbasis,tip_eigenbasis,xs,ys)
    unopt_elapsed = time.time()-start
    print("Basis Change (tip to sample): {} seconds".format(unopt_elapsed))

    threadsperblock = (TPB,TPB)
    blockspergrid_x = int(math.ceil(N_tip_eigenfunctions/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(N_sample_eigenfunctions/threadsperblock[0]))
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    d_sample_eigenbasis = cuda.to_device(sample_eigenbasis)
    d_tip_eigenbasis = cuda.to_device(tip_eigenbasis)
    d_U = cuda.device_array((N_sample_eigenfunctions,N_tip_eigenfunctions))

    start = time.time()
    ChangeBasisGPU[blockspergrid,threadsperblock](d_U,d_sample_eigenbasis,d_tip_eigenbasis,xs,ys)
    GPU_elapsed = time.time()-start
    print("Basis Change (tip to sample) with GPU: {} seconds".format(GPU_elapsed))
    print("Speedup: {}".format(unopt_elapsed/GPU_elapsed))
    U_tip_to_sample_GPU = d_U.copy_to_host()

    TestBasisChange(U_tip_to_sample,tip_eigenbasis,sample_eigenbasis)
    #TestBasisChange(U_tip_to_sample_GPU)

if __name__=='__main__': main()
