from __future__ import division
from dolfin import *
from fenics import *
from mshr import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math
import random
import FenicsTools as FT

def helmholtz(mesh, eigenvalue,number_extracted = 6, to_plot=False, **kwargs):
    """Solves the Helmholtz equation on an arbitary mesh for a specific eigenvalue. Possible to "aim" for
       a specific eigenvalue by specifying an eigenvalue. Can also extract eigenvalues near the original
       eigenvalue by specifying the number of eigenvalues expected as a return through the kwarg number_extracted.
       
        Args:
            param1: mesh, FEniCS mesh
            param2: eigenvalue, number that is the value of the specified eigenvalue
            param3: number_extracted, kwarg default set to 6, will additionally extract 6 
                    eigenvalues near the specified eigenvalue.
            param4: to_plot, when True will plot the eigenfunctions and eigenvalues

        Returns:
            Dictionary indexed by eigenvalue filled with the appropriate FEniCS function eigenfunctions

    """
    V = FunctionSpace(mesh, 'Lagrange', 3)
    Pot = Expression('1.0',element = V.ufl_element())

    #build essential boundary conditions
    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V,Constant(0.0), u0_boundary)

    #define functions
    u = TrialFunction(V)
    v = TestFunction(V)

    #define problem
    a = (inner(grad(u), grad(v)) \
         + Pot*u*v)*dx
    m = u*v*dx

    A = PETScMatrix() 
    M = PETScMatrix()
    _ = PETScVector()
    L = Constant(0.)*v*dx

    assemble_system(a, L, bc, A_tensor=A, b_tensor=_)
    #assemble_system(m, L,bc, A_tensor=M, b_tensor=_)
    assemble_system(m, L, A_tensor=M, b_tensor=_)

    #create eigensolver
    eigensolver = SLEPcEigenSolver(A,M)
    eigensolver.parameters['spectrum'] = 'target real'
    eigensolver.parameters['tolerance'] = 1.e-15
    eigensolver.parameters["spectral_transform"] = "shift-and-invert"
    eigensolver.parameters["spectral_shift"] = float(eigenvalue) # Could be a possible spot for overflow

    #solve for eigenvalues
    eigensolver.solve(number_extracted)

    assert eigensolver.get_number_converged() > 0

    eigenvalues = []
    eigenfunctions = []

    for i in range(number_extracted):
        u = Function(V)
        r, c, rx, cx = eigensolver.get_eigenpair(i)

        #assign eigenvector to function
        u.vector()[:] = rx
        eigenvalues.append(r)
        eigenfunctions.append(u)
        
        if to_plot:
            plt.figure(); plot(u,interactive=True);plt.title("Eigenvalue: {}".format(r))

    return dict(zip(eigenvalues,eigenfunctions))
    
#--------------------Sample Creation Tool-------------------#
class RectangularSample(object):
    
    def __init__(self,width,height):
        """Initializes the rectangular sample. The established coordinate system on the rectangular mesh is then determined
           as follows. 
               -The lower left corner of the sample is considered point (0,0).
               -The height is the y-axis
               -The width is the x-axis
               -All subsequent coordinate values are positive
        Args:
            param1: width, int
            param2: height, int
            
        Returns:
            RectangularSample object
        """
        self.width = width
        self.height = height
        self.domain = Rectangle(Point(0, 0),
                    Point(width, height))
        self.number_of_objects = 1
        
    def getMesh(self,density = 200):
        """Produces a Fenics Mesh Object from the rectangular sample object
        
        Args:
            param1: density, kwarg default set to 200, can modify if a denser or sparser mesh is required
            
        Returns:
            Fenics Mesh Object
        """
        return generate_mesh(self.domain, density)
    
    def placeCircularSource(self,x_pos,y_pos,sourceRadius):
        """Places a circular source, a circular region with dirichlet boundary conditions, at a specified
           coordinate position on the sample.
        
        Args:
            param1: x_pos, positive x coordinate < width
            param2: y_pos, positive y coordainte < height
            param3: sourceRadius, positive value for the radius of the circular source
            
        Returns:
            None
        """
        self.number_of_objects += 1
        circ = Circle(Point(x_pos, y_pos), sourceRadius/2)
        self.domain.set_subdomain(self.number_of_objects, circ)
        
    def placeCircularReflector(self,x_pos,y_pos,reflectorRadius):
        """Places a circular reflector, a circular region with neumann boundary conditions, at a specified
           coordinate position on the sample.
        
        Args:
            param1: x_pos, positive x coordinate < width
            param2: y_pos, positive y coordainte < height
            param3: reflectorRadius, positive value for the radius of the circular source
            
        Returns:
            None
        """
        circ = Circle(Point(x_pos, y_pos), reflectorRadius/2)
        self.domain-=circ
        
    def placeRectangularSource(self,x_pos,y_pos,width, height):
        """Places a rectangular source, a rectangular region with dirichlet boundary conditions, at a specified
           coordinate position on the sample. The x_pos and y_pos arguments specify the position of the bottom
           left corner of the rectangular source. 
        
        Args:
            param1: x_pos, positive x coordinate < self.width
            param2: y_pos, positive y coordainte < self.height
            param3: width, postive value for the width of the source
            param4: height, postive value for the height of the source
            
        Returns:
            None
            
        """
        self.number_of_objects += 1
        rec = Rectangle(Point(x_pos, y_pos),
                    Point(x_pos+width, y_pos+height))
        self.domain.set_subdomain(self.number_of_objects, rec)
        
    #the position placement is dependent on the bottom left hand corner
    def placeRectangularReflector(self,x_pos,y_pos,width, height):
        """Places a rectangular reflector, a rectangular region with neumann boundary conditions, at a specified
           coordinate position on the sample. The x_pos and y_pos arguments specify the position of the bottom
           left corner of the rectangular reflector. 
        
        Args:
            param1: x_pos, positive x coordinate < self.width
            param2: y_pos, positive y coordainte < self.height
            param3: width, postive value for the width of the reflector
            param4: height, postive value for the height of the reflector
            
        Returns:
            None
            
        """
        rec = Rectangle(Point(x_pos, y_pos),
                    Point(x_pos+width, y_pos+height))
        self.domain-=rec
        
#     def n_placeCircularReflector(self,x_pos,y_pos,reflectorRadius):
#         self.number_of_objects += 1
#         circ = Circle(Point(x_pos, y_pos), sourceRadius/2)
#         self.domain.set_subdomain(self.number_of_objects, circ)
        
        
#     def define_boundaries(self, density = 200):
#         defined_mesh = self.getMesh(density=density)
#         self.boundary_markers = MeshFunction('size_t', defined_mesh, 2, mesh.domains())
#         self.boundaries = MeshFunction('size_t', defined_mesh, 1, mesh.domains())

#         # Use the cell domains associated with each facet to set the boundary
#         for f in facets(defined_mesh):
#             domains = []
#             for c in cells(f):
#                 domains.append(self.boundary_markers[c])

#             domains = list(set(domains))
#             if len(domains) > 1:
#                 self.boundaries[f] = 2
                
    def set_boundary_conditions(self, function_space, dim = 1, density = 200):
        """Uses the generate mesh functionality in order to set boundary conditions before a run. 
        
           **A bit buggy, run already sets the boundary conditions when it goes so this code isn't
             normally used**
        
        Args:
            param1: function_space, FEniCS function space for the boundary condition to e 
            param2: dim, kwarg default set to 1, the dimension of the function space
            param3: density, kwarg default set to 200, the density of the auto-generated mesh
            
        Returns:
            None
            
        """
        defined_mesh = self.getMesh(density=density)
        self.boundary_markers = MeshFunction('size_t', defined_mesh, 2, mesh.domains())
        self.boundaries = MeshFunction('size_t', defined_mesh, 1, mesh.domains())

        # Use the cell domains associated with each facet to set the boundary
        for f in facets(defined_mesh):
            domains = []
            for c in cells(f):
                domains.append(self.boundary_markers[c])

            domains = list(set(domains))
            if len(domains) > 1:
                self.boundaries[f] = 2
        
        #assumes dim = 1
        bc_O_value = Constant(0.0,cell=defined_mesh.ufl_cell())
        bc_I_value = Constant(1.0,cell=defined_mesh.ufl_cell())
        
        #Will modify for a mixed function space
        if (dim == 2):
            bc_O_value = (bc_O_value,bc_O_value)
            bc_I_value = (bc_I_value,bc_I_value)
            
        def u0_boundary(x, on_boundary):
            return on_boundary
        
        print bc_O_value
        print bc_I_value

        bc_O = DirichletBC(function_space, bc_O_value, u0_boundary)

        bc_I = DirichletBC(function_space, bc_I_value, self.boundaries, 2)

        return [bc_O,bc_I]

    def run(self,omega,sigma, to_plot=False, density = 200):
        """Solves the plasmon wave equation on the pre-defined sample. Requires input omega and sigma objects
           that are the parameters for the plasmon wave equation.
        
        Args:
            param1: omega, object that encapsulates the kappa and V parameters of the wave equation
            param2: sigma, object that encapsulates the lambda and L parameters of the wave equation
            param3: to_plot, kwarg default = False, if True will print out what the real and imaginary
                    parts of the function look like
            param4: density, kwarg default set to 200, the density of the auto-generated mesh
            
        Returns:
            a FEniCS function that represents the solution to the wave equation
            
        """
        mesh = self.getMesh(density)
        L3 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
        V = MixedElement([L3,L3])
        ME = FunctionSpace(mesh,V)

        # Define boundaries 
        boundary_markers = MeshFunction('size_t', mesh, 2, mesh.domains())
        boundaries = MeshFunction('size_t', mesh, 1, mesh.domains())

        # Use the cell domains associated with each facet to set the boundary
        for f in facets(mesh):
            domains = []
            for c in cells(f):
                domains.append(boundary_markers[c])

            domains = list(set(domains))
            if len(domains) > 1:
                boundaries[f] = 2

        def u0_boundary(x, on_boundary):
            return on_boundary
        
        #Establishing the Boundary Conditions

        bc_O = DirichletBC(ME, (Constant(0.0),Constant(0.0)), u0_boundary)

        bc_I = DirichletBC(ME,(Constant(1.0),(Constant(1.0))), boundaries, 2)

        s_1,s_2 = sigma.get_sigma_values()
        o = omega.get_omega_values()

        #trial and test functions
        q,v = TestFunctions(ME)

        # Define functions
        u = TrialFunction(ME) # current
        z = Function(ME)

        # Split mixed functions
        u_1,u_2 = split(u)

        # u_init = InitialConditions(degree = 1)
        # u.interpolate(u_init)

        s_1 = Constant(s_1)
        s_2 = Constant(s_2)
        o = Constant(o)

        F0 = (s_1*s_1+s_2*s_2)*dot(grad(u_1), grad(q))*dx+o*(s_1*u_2-s_2*u_1)*q*dx
        F1 = (s_1*s_1+s_2*s_2)*dot(grad(u_2), grad(v))*dx-o*(s_1*u_1+s_2*u_2)*v*dx
        F = F0 + F1 + Constant(20.0)*q*ds + Constant(20.0)*v*ds
        a, L = lhs(F), rhs(F)

        print "Attempting to solve:"

        solve(a==L,z,[bc_O,bc_I])
        
        if to_plot:
            fig = plt.figure()
    #         plt.subplot(131);mplot(mesh);plt.title("Mesh"),plt.tick_params(
    #             axis='both',          # changes apply to the x-axis
    #             which='both',      # both major and minor ticks are affected
    #             bottom=False,
    #             right = False,     # ticks along the bottom edge are off
    #             left = False,
    #             top=False)         # ticks along the top edge are off)
            plt.subplot(121);mplot(z.split(deepcopy=True)[0]);plt.title("Real Part"),plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,
                right = False,     # ticks along the bottom edge are off
                left = False,
                top=False,         # ticks along the top edge are off
                labelleft=False);plt.colorbar()
            plt.subplot(122);mplot(z.split(deepcopy=True)[1]);plt.title("Im Part"),plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,
                right = False,     # ticks along the bottom edge are off
                left = False,
                top=False,         # ticks along the top edge are off
                labelleft=False);plt.colorbar()

    #         fig.subplots_adjust(right=0.8)
    #         cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #         fig.colorbar(plt.subplot(133), cax=cbar_ax)
            plt.show()
        
        self.solution = z
        
        return z
    
    def eigenvalue_solve(self,eigenvalue,number_extracted = 6, to_plot = False, density = 200):
        """Uses the SlepcEigensolver to solve the helmholtz equation on the pre-defined sample. Requires a specified eigenvalue 
           to "target". Will extract up to number_extracted nearby eigenvalue, eigenvector pairs.
        
        Args:
            param1: eigenvalue, float, the value that the SlepcEigensolver will attempt to "aim" for.
            param2: number_extracted, kwarg default set to 6, will additionally extract 6 
                    eigenvalues near the specified eigenvalue.
            param3: to_plot, kwarg default = False, if True will print out what the eigenvectors look like
                    with eigenvalue titles
            param4: density, kwarg default set to 200, the density of the auto-generated mesh
            
        Returns:
            Dictionary indexed by eigenvalue filled with the appropriate FEniCS function eigenfunctions
            
        """
        mesh = self.getMesh(density)
        V = FunctionSpace(mesh, 'Lagrange', 3)
        
        # Define boundaries 
        boundary_markers = MeshFunction('size_t', mesh, 2, mesh.domains())
        boundaries = MeshFunction('size_t', mesh, 1, mesh.domains())

        # Use the cell domains associated with each facet to set the boundary
        for f in facets(mesh):
            domains = []
            for c in cells(f):
                domains.append(boundary_markers[c])

            domains = list(set(domains))
            if len(domains) > 1:
                boundaries[f] = 2

        def u0_boundary(x, on_boundary):
            return on_boundary
        
        #Establishing the Boundary Conditions
        bc_O = DirichletBC(V, Constant(0.0), u0_boundary)

        bc_I = DirichletBC(V,Constant(1.0), boundaries, 2)

        #define functions
        u = TrialFunction(V)
        v = TestFunction(V)

        #define problem
        Pot = Expression('1.0',element = V.ufl_element())
        a = (inner(grad(u), grad(v)) \
             + Pot*u*v)*dx
        m = u*v*dx

        A = PETScMatrix() 
        M = PETScMatrix()
        _ = PETScVector()
        L = Constant(0.)*v*dx

        bcs = [bc_O,bc_I]

        assemble_system(a, L, bcs, A_tensor=A, b_tensor=_)
        #assemble_system(m, L,bc, A_tensor=M, b_tensor=_)
        assemble_system(m, L, A_tensor=M, b_tensor=_)

        #create eigensolver
        eigensolver = SLEPcEigenSolver(A,M)
        eigensolver.parameters['spectrum'] = 'target real'
        eigensolver.parameters['tolerance'] = 1.e-15
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["spectral_shift"] = float(eigenvalue) # Could be a possible spot for overflow

        #solve for eigenvalues
        eigensolver.solve(number_extracted)

        assert eigensolver.get_number_converged() > 0

        eigenvalues = []
        eigenfunctions = []
            
        #returning the eigenvalue, eigenvector pairs
        for i in range(number_extracted):
            u = Function(V)
            r, c, rx, cx = eigensolver.get_eigenpair(i)

            #assign eigenvector to function
            u.vector()[:] = rx
            eigenvalues.append(r)
            eigenfunctions.append(u)
            plt.figure(); plot(u,interactive=True); plt.title("Eigenvalue: {}".format(r))

        return dict(zip(eigenvalues,eigenfunctions))
        
    def cast_solution_to_AWA(self,density=200):
        """Casts the FEniCS function solution to an ArrayWithAxes object.
        
        Args:
            param1: density, kwarg default set to 200, the density of the mesh to interpolate onto
            
        Returns:
            ArrayWithAxes object
            
        """
        mesh0 = self.getMesh(density)
        L3 = FiniteElement("Lagrange", mesh0.ufl_cell(), 2)
        V = MixedElement([L3,L3])
        ME0 = FunctionSpace(mesh0,V)
        
        mesh1 = RectangleMesh(Point(0,0),Point(self.width,self.height),density,density)
        L3 = FiniteElement("Lagrange", mesh1.ufl_cell(), 2)
        V = MixedElement([L3,L3])
        ME1 = FunctionSpace(mesh1,V)
        
        if self.solution is None:
            print "You need to run the sample first"
            return
        
        self.solution.set_allow_extrapolation(True)
        Pv = interpolate(self.solution,ME1)
        
        V2 = FunctionSpace(mesh1, 'CG', 1)
        u0 = interpolate(Pv.split(deepcopy=True)[0], V2)
        u1 = interpolate(Pv.split(deepcopy=True)[1], V2)
        
        BF0 = FT.BoxField2(u0)
        BF1 = FT.BoxField2(u1)
        
        return (BF0.to_AWA(),BF1.to_AWA())
    
#------------------------------------------------------------------------------------------------------------------------------#
    
class S(object):
    """
        Sigma term parameters, reference equations in Rodin/accompanying manuscript
    """
    
    def set_sigma_values(self,lambda_r,L_i):
        """Determines real and imaginary parts of sigma. (Re) sigma_1 and (Im) sigma_2. 
        
        Args:
            param1: lambda_r, float or int, a number
            param2: L_i, float or int, a number
        Returns:
            None   
        """
        self.sigma_1, self.sigma_2 = (1/L_i)/((1/lambda_r)**2+(1/L_i)**2), (1/lambda_r)/((1/lambda_r)**2+(1/L_i)**2)
    
    def get_sigma_values(self):
        """Returns real and imaginary parts of sigma. (Re) sigma_1 and (Im) sigma_2. 
        
        Args:

        Returns:
            tuple, (sigma_1,sigma_2)   
        """
        return self.sigma_1,self.sigma_2
        
class O(object):
    """
        Omega term parameters, reference equations in Rodin/accompanying manuscript
    """
    
    def set_omega_values(self,kappa,V):
        """Calculates omega from input parameters kappa and V
        
        Args:
            param1: kappa, float or int, a number
            param2: V, float or int, a number
        Returns:
            None   
        """
        self.omega = (2*math.pi)/(kappa*V)
    
    def get_omega_values(self):
        """Returns omega value
        
        Args:

        Returns:
            float, omega   
        """
        return self.omega
        
#-------------------------------------------------------------------------------------------------------------------------------#

class equationPhraser(object):
    """Working on it, will phrase the parameters for solving the plasmon wave equation as a specific eigenvalue for solving the 
       eigenvalue problem on the helmholtz solver.
            
    """
    
    def set_values(self,eigenval):
        pass
        #Need some parameters to specify
        
#--------------------------------------------------------------------------------------------------------------------------------#
#Additional Fenics Functions
def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

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
            plt.tripcolor(mesh2triang(mesh), C)
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')