# BokehPlasmons
Bokeh Based Interface for the Modeling Plasmons



app.py - contains the majority of the logic for the GUI
PlacedObjects.py - definitions for the objects manipulated in the GUI
Plasmon_Modeling.py - main file that contains the logic in order to run Plasmon Simulations
Toolbox.py - collection of useful functions for interacting with FENICS
Checking Rotation and Cutting Shapes.ipynb - test notebook for developing the rotation code and demonstrating cutting shapes

# Additional Features to be added
Support for rotating samples, need to be integrated into the PlacedObjects.py but possible to perform in a notebook
Undo Button

# To run this
bokeh serve app.py --port 5100
