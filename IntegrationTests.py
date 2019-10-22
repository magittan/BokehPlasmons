import os, h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.integrate import nquad
from scipy.interpolate import interp2d

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

def IntFun(fun,x,y):
    x = int((x+1)*len(fun)/2)
    y = int((y+1)*len(fun)/2)
    return fun[x,y]

def IntTest():
    extent = [-1,1,-1,1]
    nx, ny=1001,1001
    xs = np.linspace(-1,1,nx)
    ys = np.linspace(-1,1,ny)
    xv,yv = np.meshgrid(xs,ys,sparse=True)
    rs = np.sqrt(xv**2+yv**2)
    theta = np.arctan(yv/xv)
    theta[np.where(np.isnan(theta))] = 0

    bessels = []
    q=10
    for v in range(10):
        bessels.append(jv(v,q*rs))
    #ImshowAll(np.array(bessels))
    for v in range(len(bessels)):
        quad = nquad(lambda x,y: IntFun(bessels[v],x,y),[[-1,1],[-1,1]])[0]
        trap = np.trapz(np.trapz(bessels[v],ys),xs)
        my = MyIntFunc(bessels[v],xs,ys)
        print('my: {}\nnquad: {}\ntrapz: {}\nPercent Diff: {}\n'.format(my,quad,trap,100*(quad-trap)/quad))

def MyIntFunc(f, xs, ys):
    sum = 0
    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]
    for row in f:
        for z in row:
            sum += z*dx*dy
    return sum

def MeshTest():
    xs = np.linspace(-1,1,101)
    ys = np.linspace(-1,1,101)
    fine_xs = np.linspace(-1,1,1001)
    fine_ys = np.linspace(-1,1,1001)
    dirname = './'
    fname = 'UnitSquareMesh_100x100_1000_eigenbasis.h5'
    full_fname = os.path.join(dirname, fname)
    sample_eigenvalues, sample_eigenfuncs = [],[]
    with h5py.File(full_fname,'r') as f:
        for key in list(f.keys()):
            sample_eigenvalues.append(float(key))
            sample_eigenfuncs.append(np.array(f.get(key)))
    eigenfunc = sample_eigenfuncs[3]
    interp_eigenfunc = interp2d(xs,ys,eigenfunc)

    plt.subplot(121)
    plt.imshow(eigenfunc)
    plt.subplot(122)
    plt.imshow(interp_eigenfunc(fine_xs,fine_ys))
    plt.show()

def main():
    #MeshTest()
    IntTest()

    #plt.figure(); plt.imshow(bessels[5]); plt.show()

if __name__=='__main__': main()
