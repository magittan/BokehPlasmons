import PlasmonModeling as PM
from dolfin import *
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import h5py, time, os, argparse

def set_parser():
    parser = argparse.ArgumentParser(description = "Generate an HDF5 containing sample eigenbasis for a unit square configuration produced by solving the homogeneous Helmholtz equation.",\
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--density', type=int, default=100,\
                            help="integer specifying the mesh density on x- and y- axes")
    parser.add_argument('--eigenvalue', type=float, default=5,\
                            help="float specifying the aiming eigenvalue")
    parser.add_argument('--n_extracted', type=int, default=10,\
                            help="number of eigenfunctions to be extracted")
    parser.add_argument('--wavelength', type=float, default=1,\
                            help="wavelength of the plasmon")
    parser.add_argument('--L', type=float, default=40,\
                            help="propagation length of the plasmon")
    return parser

def main():
    parser = set_parser()
    args = parser.parse_args()
    datadir = './sample_eigenbasis_data'
    show = False

    wavelength = args.wavelength
    L = args.L
    density = args.density
    eigenvalue = args.eigenvalue
    n_extracted = args.n_extracted

    sigma = PM.S()
    sigma.set_sigma_values(wavelength,L)
    s_1,s_2 = sigma.get_sigma_values()
    mesh = UnitSquareMesh(density, density)
    start = time.time()
    eigenvalue_eigenfunction_pairs = PM.helmholtz(mesh, eigenvalue,\
                                                    number_extracted=n_extracted,\
                                                    sigma_2 = s_2,\
                                                    to_plot=False)
    end = time.time()
    print('Time elapsed: {} seconds'.format(end-start))

    #Getting eigenvalue eigenfunction pairs and storing it in a dictionary
    processed_eigenvalue_eigenfunction_pairs = dict()
    for key, value in eigenvalue_eigenfunction_pairs.items():
        processed_eigenvalue_eigenfunction_pairs[key]=PM.fenics_func_to_AWA(value, mesh)

    fname = os.path.join(datadir, "UnitSquareMesh_{}x{}_{}_eigenbasis.h5".format(density,density,n_extracted))

    with h5py.File(fname,'w') as f:
        for key, value in processed_eigenvalue_eigenfunction_pairs.items():
            f[str(key)] = np.array(value)

if __name__=='__main__': main()
