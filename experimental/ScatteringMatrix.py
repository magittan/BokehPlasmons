import Plasmon_Modeling as PM
import CoulombKernel as CK
import BasisChangeTest as BCT
import numpy as np
import matplotlib.pyplot as plt
import h5py

def GetReducedEigenbasis(num_eigs, aim_eigval, sample_eigenbasis):
    index=np.argmin(np.abs(sample_eigenvalues-q_omega))
    ind1=np.max([index-num_eigs//2,0])
    ind2=ind1+num_eigs
    use_eigenvalues = sample_eigenvalues[ind1:ind2]


def GetScatteringMatrix(num_eigs, q_omega, r, qs, deg, sigma, sample_eigenbasis):
    """
        Get the scattering matrix S_{omega,sigma} (from Modeling_Plasmon_Flowchart__AM.pdf)

        Args:
        param1: mesh, FEniCS mesh
        param2: eigenvalue, number that is the value of the specified eigenvalue
        param3: number_extracted, kwarg default set to 6, will additionally extract 6
                eigenvalues near the specified eigenvalue.
        param4: to_plot, when True will plot the eigenfunctions and eigenvalues

        Returns:
        Dictionary indexed by eigenvalue filled with the appropriate FEniCS function eigenfunctions
    """

    qmin,qmax = qs
    sigma_tilde = sigma.get_sigma_values()[0]+1j*sigma.get_sigma_values()[1]
    alpha = -1j*sigma_tilde/abs(sigma_tilde)

    sample_eigenvalues = np.array(list(sorted(sample_eigenbasis.keys())))

    index=np.argmin(np.abs(sample_eigenvalues-q_omega))
    ind1=np.max([index-num_eigs//2,0])
    ind2=ind1+num_eigs
    use_eigenvalues = sample_eigenvalues[ind1:ind2]

    # Get diagonal matrix of squared eigenvalues
    Q = np.diag(use_eigenvalues[:num_eigs]**(2))

    # Get the Coulomb Kernel
    V_object = CK.V(sample_eigenbasis_file,q_omega,r,qmin,qmax,degree)
    V_nm = V_object.get_V(Q.shape[0])

    scatter = q_omega*np.linalg.inv(q_omega*np.identity(Q.shape[0]) - alpha*Q.dot(V_nm))
    return scatter

if __name__=='__main__':
    sample_eigenbasis_fname = "UnitSquareMesh_100x100_1000_eigenbasis.h5"
    sample_eigenbasis_file = h5py.File(sample_eigenbasis_fname,'r')
    sample_eigenbasis = BCT.LoadEigenbasis(sample_eigenbasis_fname)
    """
    num_eigs=100
    q_omega = 1000
    r = np.linspace(1e-7,150e-7,1000)
    qmin=1;qmax=1e10;degree=20
    L = 1000; lamb = 5

    sigma = PM.S()
    sigma.set_sigma_values(lamb, L)

    sample_eigenbasis_fname = "UnitSquareMesh_100x100_1000_eigenbasis.h5"
    sample_eigenbasis_file = h5py.File(sample_eigenbasis_fname,'r')
    sample_eigenbasis = BCT.LoadEigenbasis(sample_eigenbasis_fname)

    scatter = GetScatteringMatrix(num_eigs, q_omega, r, [qmin,qmax], degree, sigma, sample_eigenbasis)

    plt.figure()
    plt.imshow(np.log10(np.abs(scatter)))
    plt.colorbar()
    plt.show()
    """
