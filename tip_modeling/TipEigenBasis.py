import pickle
import numpy as np
from copy import copy
from common.baseclasses import AWA
from common import baseclasses
from common import plotting
from NearFieldOptics.Materials.material_types import *


class TipEigenBasis():
    """
    Wrapper class for cached values of Tip eigenbasis at sample (i.e. z=0).
    Cached values:
        Ez: E field (z-component) profile as a function of r
        Er: E field r-component
        V: scalar potential profile as a function of r
        P: "magic" reflection coefficient; or poles of polarizability
        R: residues of polarizability
    """

    def __init__(self, file_name):
        """
        Construct a TipEigenBasis object.

        Args:
            file_name (str): name of the picke file of the dictionary outputed from the EvaluateEigenfield() method in ExpansionApproximation module
        
        return: void
        """

        with open(file_name, 'rb') as handle:
            d = pickle.load(handle)

        self.Ezs = d['Ez']
        self.Ers = d['Er']
        self.Vs = d['V']
        self.Ps = d['P']
        self.Rs = d['R']
        self.max_n = self.Ezs.shape[0]

    """
    getter methods for class variables
    """
    def get_Ezs(self): return copy.copy(self.Ezs)
    def get_Ers(self): return copy.copy(self.Ers)
    def get_Vs(self): return copy.copy(self.Vs)
    def get_Ps(self): return copy.copy(self.Ps)
    def get_Rs(self): return copy.copy(self.Rs)
    def get_max_n(self): return copy.copy(self.max_n)
    
    """
    getter method for P specific given n
        n starts from 0.
    """
    def get_P_targeted(self,n):
        P = self.Ps[n]
        return copy.copy(P)
    def get_R_targeted(self,n):
        R = self.Rs[n]
        return copy.copy(R)
    
    """
    getter method for the potential profile at z=0.
    n starts from 0.
    """
    def get_V_function(self,n,ztip,z=0,**kwargs):
        """
        Return a lambda function of x,y
        """
        #check n is within interpolation range
        if n>self.max_n:
            Logger.raiseException('ztip exceed the interpolation bound: '+str(ztip_max),exception=ValueError)
        
        V_n = self.Vs.cslice[n,:,:]
        ztip_max = max(V_n.axes[0])
#         
#         imag = np.imag(V_n).flatten()
#         print('max imaginary')
#         print(max(imag))
        
        #check ztip is within interpolation range
        if ztip>ztip_max:
        	Logger.raiseException('ztip exceed the interpolation bound: '+str(ztip_max),exception=ValueError)
        
        from scipy.interpolate import interp2d
        interpolator_real=interp2d(V_n.axes[1],V_n.axes[0],np.real(V_n),**kwargs)
        interpolator_imag=interp2d(V_n.axes[1],V_n.axes[0],np.imag(V_n),**kwargs)
            
        def potential(xs,ys):
        
            from common.baseclasses import AWA
        
            x_mesh,y_mesh=np.meshgrid(xs,ys)
            r_mesh=np.sqrt(x_mesh**2+y_mesh**2)
            V_2D_real = np.zeros(shape=x_mesh.shape,dtype=np.complex)
            V_2D_imag = np.zeros(shape=x_mesh.shape,dtype=np.complex)
            i=0
            for x in xs:
                j=0
                for y in ys:
                    r = np.sqrt(x**2+y**2)
                    V_2D_real[i,j]=interpolator_real(r_mesh[i,j],ztip)
                    V_2D_imag[i,j]=interpolator_imag(r_mesh[i,j],ztip)
                    j+=1
                i+=1
            V_2D = V_2D_real+V_2D_imag*1j
            V_2D = AWA(V_2D,axes=[xs,ys],axis_names=['x (a)','y (a)'])
            
            return V_2D
        
        return potential
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
