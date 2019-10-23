import h5py,time,os, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.integrate import nquad
from numba import jit, cuda, prange


def Progress(i,L,last):
    next = last+10
    percent = 100*i/L
    if percent >= next:
        print('{}% complete...'.format(next))
        return next
    else:
        return last

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
            last = Progress(to_n*from_n, N, last)
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

def ShowSampleAndTip(sample_m,tip_m,origin='lower',extent=None):
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

def VInBasis(vec,basis):
    basis_shape = basis[0].shape
    sum = np.zeros(basis_shape)
    for i,vi in enumerate(vec):
        sum += vi*basis[i]
    return sum

def ConstructEigenbases(N_tip_eigenfunctions,N_sample_eigenfunctions,rs,theta,sample_eigenvalues,sample_eigenpairs):
    tip_eigenbasis, sample_eigenbasis = [],[]

    for n in range(N_sample_eigenfunctions):
        sample_eigenbasis.append(sample_eigenpairs[sample_eigenvalues[n]])

    for alpha in range(N_tip_eigenfunctions):
        tip_eigenbasis.append(jv(alpha//10,10*rs)*np.cos((alpha%10)*theta))

    sample_eigenbasis = np.asarray(sample_eigenbasis)
    tip_eigenbasis = np.asarray(tip_eigenbasis)
    return tip_eigenbasis,sample_eigenbasis

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

def OneHot(i,L):
    onehot = np.concatenate((np.array([1]),np.zeros(L-1)))
    onehot = np.roll(onehot,i)
    return onehot

def main():
    sample_eigenpairs = LoadEigenbasis('UnitSquareMesh_100x100_1000_eigenbasis.h5')
    sample_eigenvalues = sorted(list(sample_eigenpairs.keys()))
    TPB = 16

    extent = [-1,1,-1,1]

    nx, ny=101,101
    xs, ys = np.linspace(-1,1,nx), np.linspace(-1,1,ny)
    xv,yv = np.meshgrid(xs,ys,sparse=True)

    rs = np.sqrt(xv**2+yv**2)
    theta = np.arctan(yv/xv)
    theta[np.where(np.isnan(theta))] = 0
    times = {}

    #for n in range(200):
    for N_tip_eigenfunctions in [100]:
        for N_sample_eigenfunctions in [1000]:
            # Construct tip and sample eigenbases
            tip_eigenbasis, sample_eigenbasis = ConstructEigenbases(N_tip_eigenfunctions,N_sample_eigenfunctions,rs,theta,sample_eigenvalues,sample_eigenpairs)
            #ImshowAll(tip_eigenbasis)

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
            res = d_U.copy_to_host()
            #ShowSampleAndTip(U_tip_to_sample,res)
            times[N_tip_eigenfunctions*N_sample_eigenfunctions] = unopt_elapsed/GPU_elapsed

    tip_v = OneHot(0,N_tip_eigenfunctions)+OneHot(1,N_tip_eigenfunctions)+OneHot(2,N_tip_eigenfunctions)
    tip_m = VInBasis(tip_v,tip_eigenbasis)
    sample_v = U_tip_to_sample.dot(tip_v.T)
    sample_v_GPU = res.dot(tip_v.T)
    sample_m = VInBasis(sample_v,sample_eigenbasis)
    sample_m_GPU = VInBasis(sample_v_GPU,sample_eigenbasis)
    """
    sample_v = OneHot(0,N_sample_eigenfunctions)+OneHot(1,N_sample_eigenfunctions)
    sample_m = VInBasis(sample_v,sample_eigenbasis)
    tip_v = U_sample_to_tip.dot(sample_v.T)
    tip_m = VInBasis(tip_v,tip_eigenbasis)

    sample_m = sample_m/np.max(sample_m)
    sample_m_GPU = sample_m_GPU/np.max(sample_m_GPU)
    tip_m = tip_m/np.max(tip_m)
    diff = np.sum((sample_m-tip_m)**2)/np.prod(sample_m.shape)
    res_norm = res /np.max(res)
    U_tip_to_sample_norm = U_tip_to_sample /np.max(U_tip_to_sample)
    diff = np.sum((res_norm-U_tip_to_sample_norm)**2)/np.prod(res_norm.shape)
    print('Difference: {}'.format(diff))
    """

    #ShowSampleAndTip(U_tip_to_sample[:10,:10],res[:10,:10],origin='upper')
    ShowSampleAndTip(sample_m_GPU,tip_m)
    ShowSampleAndTip(sample_m,tip_m)

if __name__=='__main__': main()
