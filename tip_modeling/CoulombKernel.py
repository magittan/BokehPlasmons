#!/usr/bin/env python
# coding: utf-8
from NearFieldOptics.Materials import *
from NearFieldOptics.Materials.TransferMatrixMedia import *
from common.baseclasses import AWA
from common import numerical_recipes as numrec
from sys import stdout

import h5py,operator,sympy
import scipy.special as ss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
sympy.init_printing(use_unicode=True)
mpl.rcParams['figure.dpi']= 100

def myfunc(kx,ky):
    return 1/np.sqrt(kx**2+ky**2+1e-9)

def myfunc2(x,y):
    return 1/np.sqrt(x**2+y**2+1e-4)

class V:
    def __init__(self,eigen_file,freq,r,qmin,qmax,degree):
        self.layers = LayeredMedia(SingleLayerGraphene())
        self.T = MatrixBuilder.TransferMatrix(self.layers,polarization='p')
        self.C = Calculator.Calculator(self.T)
        self.C.assemble_analytical_kernel(1,'after')

        # Quadrature nodes & weights
        #qArray,weightArray = self.get_cached_nodes_and_weights(qmin, qmax, degree)
        self.eigen_file = eigen_file

    """
    def get_cached_nodes_and_weights(self,qmin, qmax, degree):
        node = np.loadtxt('nodeArray_deg{}.txt'.format(degree),dtype=float, delimiter=' ')
        qArray = (node+1)*(qmax-qmin)/2.+qmin
        weightArray = np.loadtxt('weightArray_deg{}.txt'.format(degree),dtype=float, delimiter=' ')

        return qArray, weightArray

    def real_space_kernel(self,xRange,yRange):
        xRange = xRange.flatten()
        yRange = yRange.flatten()
        k_profile = np.empty([len(xRange), len(yRange)])
        i=0;j=0
        for x in xRange:
            j=0
            for y in yRange:
                k_profile[i][j] = self.kr_standard.cslice[np.sqrt(x**2+y**2)]
                j=j+1
            i=i+1
        return k_profile
    """

    # This function return the normalized and sorted eigenvalue-eigenfunction pairs inputed as "eigen_file" parameter
    def get_processed_eigen_pairs(self):
        eigen_pairs = dict()
        with self.eigen_file as f:
            for key in list(f.keys()):
                eigen_pairs[float(key)] = np.array(f.get(key))

        self.eigen_pairs_norm_sorted = self._normalize_and_sort_(eigen_pairs)
        return self.eigen_pairs_norm_sorted

    def _normalize_and_sort_(self,eigen_pairs):
        eigen_pairs_norm = dict()
        for item in eigen_pairs.items():
            psi = item[1]
            eigen_pairs_norm[item[0]] = item[1]/np.sqrt(np.sum(psi*psi))
        eigen_pairs_norm_sorted = sorted(eigen_pairs_norm.items(), key=operator.itemgetter(0))
        return eigen_pairs_norm_sorted

    #The MOST IMPORTANT function; the only function user call to get the Coulomb matrix
    def get_V(self,N):
        V_nm = np.zeros([N,N])
        eigen_pairs_norm_sorted = self.get_processed_eigen_pairs()
        print('QuickConv')
        freq = 1000
        # TODO: Want to give kernel_function_fourier = get_numerical_kernel
        #kernel_function_fourier=lambda kx,ky : self.C.get_numerical_kernel(freq,np.sqrt(kx**2+ky**2+1e-5))
        #myQC=numrec.QuickConvolver(size=(100,100),kernel_function_fourier=myfunc,shape=eigen_pairs_norm_sorted[0][1].shape,pad_by=.5,pad_with=0)
        myQC=numrec.QuickConvolver(size=(100,100),kernel_function=myfunc2,shape=eigen_pairs_norm_sorted[0][1].shape,pad_by=.5,pad_with=0)

        print('Loopin')
        for n in range(0,N):
            print('({})'.format(n))
            for m in range(0,N):
                psi = eigen_pairs_norm_sorted[m][1]
                psi_star = eigen_pairs_norm_sorted[n][1]
                V_nm[n,m]=np.sum(psi_star*myQC(psi))
        return V_nm

if __name__=='__main__':
    # Example Usage
    freq = 1000
    r = np.linspace(1e-7,150e-7,1000)
    qmin=1;qmax=1e10;degree=20
    eigen_file = h5py.File("UnitSquareMesh_100x100_1000_eigenbasis.h5",'r')

    V_object = V(eigen_file,freq,r,qmin,qmax,degree)
    V_nm = V_object.get_V(100,100)

    plt.imshow(np.log(abs(V_nm)))
    plt.colorbar()
    plt.title('Absolute Value of Coulomb Kernel')
    plt.show()

    #if __name__=='__main__': main()
