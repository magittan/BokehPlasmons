import h5py,time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.integrate import nquad
from numba import jit, prange

@jit
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

def ImshowAll(eigenbasis):
    N = eigenbasis.shape[0]
    fig,ax = plt.subplots(1,N)
    for i in range(N):
        ax[i].imshow(eigenbasis[i],origin='lower',extent=[-1,1,-1,1])
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

def main():
    processed_eigenvalue_eigenfunction_pairs = {}
    with h5py.File("processed_eigenvalue_eigenfunction_pairs.h5",'r') as f:
        for key in list(f.keys()):
            processed_eigenvalue_eigenfunction_pairs[float(key)] = np.array(f.get(key))
    keys = sorted(list(processed_eigenvalue_eigenfunction_pairs.keys()))

    extent = [-1,1,-1,1]
    nx,ny=101,101
    xs = np.linspace(-1,1,nx)
    ys = np.linspace(-1,1,ny)
    xv,yv = np.meshgrid(xs,ys,sparse=True)
    rs = np.sqrt(xv**2+yv**2)
    theta = np.arctan(yv/xv)
    theta[np.where(np.isnan(theta))] = 0
    N = 5 # Dimension of tip eigenbasis; in this case we use bessel functions
    # Construct tip and sample eigenbases
    nn = range(len(keys))
    aalpha = range(N)

    tip_eigenbasis = []
    sample_eigenbasis = []
    for n in nn:
        sample_eigenbasis.append(processed_eigenvalue_eigenfunction_pairs[keys[n]])
    for alpha in aalpha:
        tip_eigenbasis.append(jv(alpha//4,10*rs)*np.cos((alpha%4)*theta))
    sample_eigenbasis = np.asarray(sample_eigenbasis)
    tip_eigenbasis = np.asarray(tip_eigenbasis)
    #ImshowAll(tip_eigenbasis)

    # Construct change of basis matrix U
    U_sample_to_tip = ChangeBasis(tip_eigenbasis,sample_eigenbasis,xs,ys)
    U_tip_to_sample = ChangeBasis(sample_eigenbasis,tip_eigenbasis,xs,ys)

    """
    sample_v = np.asarray([1,0,0,0,0,0,0,0,0,0])
    sample_m = VInBasis(sample_v,sample_eigenbasis)
    tip_v = U.dot(sample_v.T)
    tip_m = VInBasis(tip_v,tip_eigenbasis)
    """


    tip_v = np.asarray([1,0,0,0,0])
    tip_m = VInBasis(tip_v,tip_eigenbasis)
    sample_v = U_tip_to_sample.dot(tip_v.T)
    sample_m = VInBasis(sample_v,sample_eigenbasis)
    print(sample_v)

    plt.figure()
    plt.subplot(121)
    plt.imshow(sample_m,origin='lower',extent=extent)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(tip_m,origin='lower', extent = extent)
    plt.colorbar()
    plt.show()

if __name__=='__main__': main()
