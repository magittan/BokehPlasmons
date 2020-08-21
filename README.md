# BokehPlasmons
Bokeh Based Interface for the Modeling Plasmons

app.py - contains the majority of the logic for the GUI
placed_objects.py - definitions for the objects manipulated in the GUI
plasmon_modeling.py - main file that contains the logic in order to run Plasmon Simulations
toolbox.py - collection of useful functions for interacting with FENICS

# Additional Features to be added

# To run this
bokeh serve app.py --port 5100

# To run this on a VM
bokeh serve --show --allow-websocket-origin='*' myapp.py

# Things to add / check
-Exportability for meshs (ability to generate meshes)

# Environment File
Conda environment saved in the environment.yml file
