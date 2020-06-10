from dolfin import *
from fenics import *
from mshr import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math
import random
import numpy as np

#------------------------------------------------Plotting and Extraction Tools-----------------------------------------------------#

def mesh2triang(mesh):
    """
    Helper Method for mplot, process_fenics_eigenfunction, etc.
    """
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def process_fenics_function(mesh,fenics_function, x_axis=np.linspace(0,1,100), y_axis=np.linspace(0,1,100)\
                                 ,to_plot=True):
    """
    Method in order to convert fenics functions into np arrays, in order to be able to deal with objects of arbitrary dimension
    the outputs will be masked Fenics Functions. This will only apply to 2D functions that need to be interpolated.
    This is build on top of the matplotlib.tri module and is effective at interpolating from non-uniform meshs.

    Args:
        param1: mesh that the fenics function is defined on
        param2: fenics function that needs to be converted into a
        param3: x_axis (np.array) needs to be a 1-D array which represents the units of the x-axis
        param4: y_axis (np.array) needs to be a 1-D array which represents the units of the y-axis
        param5: to_plot (bool) determines if the image of the interpolated function should be plotted
    Returns:
        Masked Numpy Array with the required eigenfunction
    """
    V = FunctionSpace(mesh, 'CG', 1)
    fenics_function.set_allow_extrapolation(True)
    fenics_function = interpolate(fenics_function, V)

    C = fenics_function.compute_vertex_values(mesh)

    xv, yv = np.meshgrid(x_axis, y_axis)

    test = tri.LinearTriInterpolator(mesh2triang(mesh),C)(xv,yv)

    if to_plot:
        plt.imshow(test,cmap='seismic', origin="lower")
        plt.show()

    return test

def mplot(obj):
    """Plots fenics functions in matplotlib.pyplot

        Args:
            param1: FEniCS function

        Returns:
            None
    """
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C,cmap='seismic')
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud',cmap='seismic')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')

def plot_fenics(z):
    fig = plt.figure()
    #         plt.subplot(131);mplot(mesh);plt.title("Mesh"),plt.tick_params(
    #             axis='both',          # changes apply to the x-axis
    #             which='both',      # both major and minor ticks are affected
    #             bottom=False,
    #             right = False,     # ticks along the bottom edge are off
    #             left = False,
    #             top=False)         # ticks along the top edge are off)
    real = z.split(deepcopy=True)[0]
    r_lim = max(abs(max(real.vector())),abs(min(real.vector())))
    plt.subplot(121);mplot(real);plt.title("Real Part"),plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,
        right = False,     # ticks along the bottom edge are off
        left = False,
        top=False,         # ticks along the top edge are off
        labelleft=False);
    #cax = make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05);plt.colorbar(cax=cax);plt.clim(-r_lim,r_lim)


    imaginary = z.split(deepcopy=True)[1]
    #i_lim = max(abs(max(imaginary.vector())),abs(min(imaginary.vector())))
    plt.subplot(122);mplot(imaginary);plt.title("Im Part"),plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,
        right = False,     # ticks along the bottom edge are off
        left = False,
        top=False,         # ticks along the top edge are off
        labelleft=False);
    #cax = make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05);#plt.colorbar(cax=cax);plt.clim(-i_lim,i_lim)

    #         fig.subplots_adjust(right=0.8)
    #         cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #         fig.colorbar(plt.subplot(133), cax=cbar_ax)
    plt.show()