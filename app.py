from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Column
from bokeh.io import curdoc
from bokeh.events import DoubleTap
from bokeh.models import Button, ContinuousColorMapper, BasicTicker, ColorBar
from bokeh.models.widgets import Slider
from bokeh.layouts import column, row, widgetbox
from bokeh.models.widgets import PreText, RadioButtonGroup, TextInput
from bokeh.layouts import gridplot
import Plasmon_Modeling as PM
import PlacedObjects as PO
import numpy as np


#Add sources and reflectors to certain clicked points
def DoubleClickCallback(event):
    coordinates = (event.x,event.y)
    type = details.data['type'][0]
    shape = details.data['shape'][0]

    print(coordinates)
    print(type)
    print(shape)

    #Checking the shape
    if shape == "Rectangle":
        angle = rotation.value
        width = rectangular_width.value
        height = rectangular_height.value
        input_object = PO.RectangularObject(coordinates[0]-width/2,coordinates[1]-height/2, width, height, angle)
        input_object.set_type(type)
        AddRectangle(input_object, objectList)

    elif shape == "Circle":
        input_object = PO.CircularObject(coordinates[0],coordinates[1],circular_radius.value)
        input_object.set_type(type)
        AddCircle(input_object, objectList)

def AddRectangle(input_object, in_objectList):
    print("Adding rectangle")
    search_type = input_object.get_type()
    in_objectList.append(input_object)

    temp = QueryObjectList(search_type,"Rectangle",in_objectList)
    xs = [[j[0] for j in i.get_coordinates()] for i in temp]
    ys = [[j[1] for j in i.get_coordinates()] for i in temp]

    update_data = dict(x = xs, y=ys)

    if search_type == "Source":
        p_source.data = update_data
    elif search_type == "Reflector":
        p_reflector.data = update_data

def AddCircle(input_object, in_objectList):
    search_type = input_object.get_type()
    in_objectList.append(input_object)

    temp = QueryObjectList(search_type,"Circle",in_objectList)
    x = np.array([i.get_x_coord() for i in temp])
    y = np.array([i.get_y_coord() for i in temp])
    r = np.array([i.get_radius() for i in temp])

    update_data = dict(x=x ,y=y ,r=r)

    if search_type == "Source":
        c_source.data = update_data
    elif search_type == "Reflector":
        c_reflector.data = update_data

def undo_place(in_objectList):
    search_type = in_objectList[-1].get_type()
    search_shape = in_objectList[-1].get_shape()
    in_objectList = in_objectList[:-1]

    if search_shape == "Rectangle":
        temp = QueryObjectList(search_type,"Rectangle",in_objectList)
        xs = [[j[0] for j in i.get_coordinates()] for i in temp]
        ys = [[j[1] for j in i.get_coordinates()] for i in temp]

        update_data = dict(x = xs, y=ys)

        if search_type == "Source":
            p_source.data = update_data
        elif search_type == "Reflector":
            p_reflector.data = update_data

    elif search_shape == "Circle":
        temp = QueryObjectList(search_type,"Circle",in_objectList)
        x = np.array([i.get_x_coord() for i in temp])
        y = np.array([i.get_y_coord() for i in temp])
        r = np.array([i.get_radius() for i in temp])

        update_data = dict(x=x ,y=y ,r=r)

        if search_type == "Source":
            c_source.data = update_data
        elif search_type == "Reflector":
            c_reflector.data = update_data

def QueryObjectList(in_type, in_shape, objectList):
    return [i for i in objectList if (i.get_type()==in_type) and (i.get_shape()==in_shape)]

def UpdateUpdates(input_string):
    update_messages.append(input_string)
    update_message = update_section_title
    for message in update_messages[-7:]:
        update_message+='\n'
        update_message+=message
    updates_pretext.text = str(update_message)

def update_image(data):
    new_data = dict()
    new_data['data'] = [data]
    display_data.data = new_data

#-----------------Toggling Between Different Choices-------------------#
def SetPlaceReflector():
    new_dict = {}
    new_dict['type'] = ["Reflector"]
    new_dict['shape'] = details.data['shape']
    details.data = new_dict

def SetPlaceSource():
    new_dict = {}
    new_dict['type'] = ["Source"]
    new_dict['shape'] = details.data['shape']
    details.data = new_dict

def SetPlaceCircle():
    new_dict = {}
    new_dict['type'] = details.data['type']
    new_dict['shape'] = ["Circle"]
    details.data = new_dict

def SetPlaceRectangle():
    new_dict = {}
    new_dict['type'] = details.data['type']
    new_dict['shape'] = ["Rectangle"]
    details.data = new_dict

def UpdateShapeToggle():
    indicator = toggle_circle_rectangle.active
    if indicator == 0:
        SetPlaceCircle()
    if indicator == 1:
        SetPlaceRectangle()

def UpdateTypeToggle():
    indicator = toggle_source_reflector.active
    if indicator == 0:
        SetPlaceSource()
    if indicator == 1:
        SetPlaceReflector()

#--Setting Variables----#

def SetQ(attr, old, new):
    _SetQ(new)

def SetL(attr, old, new):
    _SetL(new)

def _SetQ(new):
    new_dict = {}
    new_dict['l'] = simulation_params.data['l']
    new_dict['Q'] = [new]
    simulation_params.data = new_dict

def _SetL(new):
    new_dict = {}
    new_dict['Q'] = simulation_params.data['Q']
    new_dict['l'] = [new]
    simulation_params.data = new_dict

#-----------------------#

#----Simulating---------#
def Run():
    SimulateModel(objectList)

def Reset():
    del objectList[:]
    del update_messages[:]

    display_data.data = dict(data = [np.random.randn(50,50)])
    c_source.data = dict(x=[], y=[], r = [])
    c_reflector.data = dict(x=[], y=[], r = [])
    r_source.data = dict(top=[], bottom=[], left = [], right = [])
    r_reflector.data = dict(top=[], bottom=[], left = [], right = [])

def SimulateModel(in_objectList):
    #so far there are no checks to see if there is an error or intersection
    #source data
    c_a_r = QueryObjectList("Reflector", "Circle", in_objectList)
    c_a_s = QueryObjectList("Source", "Circle", in_objectList)
    r_a_r = QueryObjectList("Reflector","Rectangle",in_objectList)
    r_a_s = QueryObjectList("Source","Rectangle",in_objectList)

    sample = PM.RectangularSample(50,50)

    UpdateUpdates('Building Model')
    #placing reflectors
    for i in c_a_r:
        sample.placeCircularReflector(i.get_x_coord(),i.get_y_coord(),i.get_radius())

    for i in r_a_r:
        sample.placeRectangularReflector(i.get_x_coord(), i.get_y_coord(), i.get_width(), i.get_height(), i.get_angle())

    #placing sources
    for i in c_a_s:
        sample.placeCircularSource(i.get_x_coord(),i.get_y_coord(),i.get_radius())

    for i in r_a_s:
        sample.placeRectangularSource(i.get_x_coord(), i.get_y_coord(), i.get_width(), i.get_height(), i.get_angle())

    UpdateUpdates('Setting Parameters')
    #setting arbitrary omega and sigma values
    sigma = PM.S()
    omega = PM.O()
    sigma.set_sigma_values_RQ(float(simulation_params.data['l'][0]),float(simulation_params.data['Q'][0]))
    omega.set_omega_values(1,1)

    UpdateUpdates('Running Simulation')
    #running the simulation
    sample.run(omega,sigma,density = int(mesh_density_input.value))

    UpdateUpdates('Done!')
    results = sample.cast_solution_to_Array()

    display_data.data = dict(data = [results[0]])

def GenerateMesh():
    sample = PM.RectangularSample(50,50)
    density = int(mesh_density_input.value)
    sample.getMesh(density = density, to_plot=True)


# Data Sources
bound = 50
objectList= []
display_data= ColumnDataSource({'data' : [np.zeros((50,50))]})
details = ColumnDataSource({'type' : ["Source"], 'shape': ["Rectangle"],'rotation': [0]})
simulation_params = ColumnDataSource({'Q' : [100], 'l': [4]})
c_source = ColumnDataSource(data=dict(x=[], y=[], r = []))
c_reflector = ColumnDataSource(data=dict(x=[], y=[], r = []))
p_source = ColumnDataSource(data=dict(x=[],y=[]))
p_reflector = ColumnDataSource(data=dict(x=[],y=[]))

# Update message section
update_section_title = "Update Section"
update_messages = []

# Displays
clickable_display = figure(title='Double click to leave a dot.',
           tools="tap,reset",width=700,height=700,
           x_range=(0, bound), y_range=(0, bound))
output_display = figure(title='Display', plot_width=700, plot_height=700,
    x_range=(0, bound), y_range=(0, bound), tools='wheel_zoom,box_select,reset')

# Labels
updates_pretext = PreText(text='Update Section', width=350)
circular_pretext = PreText(text='Circular Reflectors/Sources', width=350)
rectangular_pretext =  PreText(text='Rectangular Reflectors/Sources', width=350)
rotation_pretext =  PreText(text='Rotation (Degrees)', width=350)
simulation_params_pretext =  PreText(text='Rotation (Degrees)', width=350)

# Sliders
circular_radius = Slider(title = "Radius",value = 2, start = 1, end = 10, step = 1)
rectangular_width = Slider(title = "Width",value = 8, start = 1, end = 10, step = 1)
rectangular_height = Slider(title = "Height",value = 2, start = 1, end = 10, step = 1)
rotation = Slider(title = "Rotation (Degrees)",value = 0, start = 0, end = 360, step = 15)

# Text Input
quality_factor_input = TextInput(value="100", title="Quality Factor")
plasmon_wavelength_input = TextInput(value="4", title="Plasmon Wavelength")
mesh_density_input = TextInput(value="100", title="Mesh Density")

# Buttons
run_button = Button(label="Run Simulation")
reset_button = Button(label="Reset Board")
generate_mesh_button = Button(label="Generate Mesh")

# Radio buttons
toggle_source_reflector = RadioButtonGroup(labels=["Source", "Reflector"], active=0)
toggle_circle_rectangle = RadioButtonGroup(labels=["Circle", "Rectangle"], active=1)

# Callbacks
run_button.on_click(Run)
reset_button.on_click(Reset)
generate_mesh_button.on_click(GenerateMesh)
toggle_source_reflector.on_change('active', lambda attr ,old, new: UpdateTypeToggle())
toggle_circle_rectangle.on_change('active', lambda attr ,old, new: UpdateShapeToggle())
quality_factor_input.on_change("value",SetQ)
plasmon_wavelength_input.on_change("value",SetL)
clickable_display.on_event(DoubleTap, DoubleClickCallback)
clickable_display.circle(source=c_source,x='x',y='y',radius = 'r', color="navy", alpha = 0.5)
clickable_display.circle(source=c_reflector,x='x',y='y',radius = 'r', color = "red", alpha = 0.5)
clickable_display.patches(source = p_source,xs='x',ys='y',color ='navy', alpha = 0.5)
clickable_display.patches(source = p_reflector,xs='x',ys='y',color ='red', alpha = 0.5)
output_display.image('data', source=display_data, palette="Viridis256",x=0, y=0, dw=bound, dh=bound)

# Set up main app
button_display = column(run_button, reset_button, generate_mesh_button, toggle_source_reflector,toggle_circle_rectangle, width = 200)
circular_slider_display = column(circular_pretext,circular_radius)
rectangular_slider_display = column(rectangular_pretext, rectangular_width, rectangular_height)
rotation_display = column(rotation_pretext,rotation)
simulation_params_display = column(quality_factor_input,plasmon_wavelength_input,mesh_density_input)

right_display = column(button_display,circular_slider_display,rectangular_slider_display,rotation_display,simulation_params_display,updates_pretext)
curdoc().add_root(row(clickable_display,output_display,right_display))
