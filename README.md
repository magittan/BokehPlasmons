# Bokeh Plasmons
Application for modeling real-space interference fringes produced by the interaction between a plasmonically active material, plasmonic sources/reflectors, and a near-field scanning probe.

The front-end is implemented using [Bokeh](https://docs.bokeh.org/en/latest/index.html) framework. The back-end uses a combination of our own implementation of the physical theory behind plasmon propagation in a sample and [FeniCS Project](https://fenicsproject.org/) finite element solver.

The underlying theory for this implementation can be found [here](../blob/master/Modeling_Plasmons_Flowchart_AM_200107.pdf).

# Installation & Setup
Use of a conda environment specifically built for this application is highly recommended. Assuming conda has been installed, the following command will set up the environment:
'''console
user@host:~$ conda create -n fenicsproject -c conda-forge fenics=2018 mshr=2018 matplotlib scipy h5py numba
'''

# Serving the Application
bokeh serve --show myapp
