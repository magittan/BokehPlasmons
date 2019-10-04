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

#setting up data
bound = 50
objectList= []
display_data= ColumnDataSource({'data' : [np.zeros((50,50))]})
details = ColumnDataSource({'type' : ["Source"], 'shape': ["Circle"],'rotation': [0]})
simulation_params = ColumnDataSource({'Q' : [100], 'l': [4]})

#Update message section
update_section_title = "Update Section"
update_messages = []

#Clickable_Display
clickable_display = figure(title='Double click to leave a dot.',
           tools="tap,reset",width=700,height=700,
           x_range=(0, bound), y_range=(0, bound))

#Output_Display
output_display = figure(title='Display', plot_width=700, plot_height=700,
           x_range=(0, bound), y_range=(0, bound), tools='pan,wheel_zoom,box_select,reset')

output_display.image('data', source=display_data, palette="Viridis256",
         x=0, y=0, dw=bound, dh=bound)

#text items
updates = PreText(text='Update Section', width=350)
circular_title = PreText(text='Circular Reflectors/Sources', width=350)
rectangular_title =  PreText(text='Rectangular Reflectors/Sources', width=350)
rotation_title =  PreText(text='Rotation (Degrees)', width=350)
simulation_params_title =  PreText(text='Rotation (Degrees)', width=350)

#setting up sliders
circular_radius = Slider(title = "Radius",value = 2, start = 1, end = 10, step = 1)
rectangular_width = Slider(title = "Width",value = 2, start = 1, end = 10, step = 1)
rectangular_height = Slider(title = "Height",value = 2, start = 1, end = 10, step = 1)
rotation = Slider(title="Rotation Degrees",value = 0,start = 0, end = 360, step = 15)

#setting up the source and reflector indicators for circles
c_source = ColumnDataSource(data=dict(x=[], y=[], r = []))
clickable_display.circle(source=c_source,x='x',y='y',radius = 'r', color="navy", alpha = 0.5)

c_reflector = ColumnDataSource(data=dict(x=[], y=[], r = []))
clickable_display.circle(source=c_reflector,x='x',y='y',radius = 'r', color = "red", alpha = 0.5)

#setting up the source and reflector indicators for rectangles
# r_source = ColumnDataSource(data=dict(top=[], bottom=[], left = [], right = []))
# clickable_display.quad(source=r_source,top='top',bottom='bottom',left = 'left', right = "right", color="navy", alpha = 0.5)
#
# r_reflector = ColumnDataSource(data=dict(top=[], bottom=[], left = [], right = []))
# clickable_display.quad(source=r_reflector,top='top',bottom='bottom',left = 'left', right = "right", color = "red", alpha = 0.5)

p_source = ColumnDataSource(data=dict(x=[],y=[]))
clickable_display.patches(source = p_source,xs='x',ys='y',color ='navy', alpha = 0.5)

p_reflector = ColumnDataSource(data=dict(x=[],y=[]))
clickable_display.patches(source = p_reflector,xs='x',ys='y',color ='red', alpha = 0.5)

#Add sources and reflectors to certain clicked points
def callback(event):
    coordinates = (event.x,event.y)
    type = details.data['type'][0]
    shape = details.data['shape'][0]

    print(coordinates)
    print(type)
    print(shape)

    #Checking the shape
    if shape == "Rectangle":
        input_object = PO.RectangularObject(coordinates[0],coordinates[1],rectangular_width.value, rectangular_height.value)
        input_object.set_type(type)
        add_rectangle(input_object, objectList)

    elif shape == "Circle":
        input_object = PO.CircularObject(coordinates[0],coordinates[1],circular_radius.value)
        input_object.set_type(type)
        add_circle(input_object, objectList)

def to_patches(data_dict):
    x_outputs = []
    y_outputs = []
    top = data_dict['top']
    bot = data_dict['bottom']
    left = data_dict['left']
    right = data_dict['right']

    for i in range(len(top)):
        temp_x = [left[i],left[i],right[i],right[i]]
        temp_y = [bot[i],top[i],top[i],bot[i]]
        x_outputs.append(temp_x)
        y_outputs.append(temp_y)

    output_dict = dict()
    output_dict['x'] = x_outputs
    output_dict['y'] = y_outputs

    return output_dict

# def add_rectangle(input_object, in_objectList):
#     print "Adding rectangle"
#     search_type = input_object.get_type()
#     in_objectList.append(input_object)
#
#     temp = query_object_list(search_type,"Rectangle",in_objectList)
#     x = np.array([i.get_x_coord() for i in temp])
#     y = np.array([i.get_y_coord() for i in temp])
#     w = np.array([i.get_width() for i in temp])
#     h = np.array([i.get_height() for i in temp])
#
#     top = y+h
#     bottom = y
#     left = x
#     right = x+w
#
#     update_data = dict(top = top, bottom = bottom, left = left, right = right)
#
#     if search_type == "Source":
#         p_source.data = to_patches(update_data)
#     elif search_type == "Reflector":
#         p_reflector.data = to_patches(update_data)

def add_rectangle(input_object, in_objectList):
    print("Adding rectangle")
    search_type = input_object.get_type()
    in_objectList.append(input_object)

    temp = query_object_list(search_type,"Rectangle",in_objectList)
    xs = [[j[0] for j in i.get_coordinates()] for i in temp]
    ys = [[j[1] for j in i.get_coordinates()] for i in temp]

    update_data = dict(x = xs, y=ys)

    if search_type == "Source":
        p_source.data = update_data
    elif search_type == "Reflector":
        p_reflector.data = update_data

# def add_rectangle(input_object, in_objectList):
#     search_type = input_object.get_type()
#     in_objectList.append(input_object)
#
#     temp = query_object_list(search_type,"Rectangle",in_objectList)
#     x = np.array([i.get_x_coord() for i in temp])
#     y = np.array([i.get_y_coord() for i in temp])
#     w = np.array([i.get_width() for i in temp])
#     h = np.array([i.get_height() for i in temp])
#
#     top = y+h
#     bottom = y
#     left = x
#     right = x+w
#
#     update_data = dict(top = top, bottom = bottom, left = left, right = right)
#
#     if search_type == "Source":
#         r_source.data = update_data
#     elif search_type == "Reflector":
#         r_reflector.data = update_data

def add_circle(input_object, in_objectList):
    search_type = input_object.get_type()
    in_objectList.append(input_object)

    temp = query_object_list(search_type,"Circle",in_objectList)
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
        temp = query_object_list(search_type,"Rectangle",in_objectList)
        xs = [[j[0] for j in i.get_coordinates()] for i in temp]
        ys = [[j[1] for j in i.get_coordinates()] for i in temp]

        update_data = dict(x = xs, y=ys)

        if search_type == "Source":
            p_source.data = update_data
        elif search_type == "Reflector":
            p_reflector.data = update_data

    elif search_shape == "Circle":
        temp = query_object_list(search_type,"Circle",in_objectList)
        x = np.array([i.get_x_coord() for i in temp])
        y = np.array([i.get_y_coord() for i in temp])
        r = np.array([i.get_radius() for i in temp])

        update_data = dict(x=x ,y=y ,r=r)

        if search_type == "Source":
            c_source.data = update_data
        elif search_type == "Reflector":
            c_reflector.data = update_data

def query_object_list(in_type, in_shape, objectList):
    return [i for i in objectList if (i.get_type()==in_type) and (i.get_shape()==in_shape)]

def reset():
    del objectList[:]
    del update_messages[:]

    display_data.data = dict(data = [np.random.randn(50,50)])
    c_source.data = dict(x=[], y=[], r = [])
    c_reflector.data = dict(x=[], y=[], r = [])
    r_source.data = dict(top=[], bottom=[], left = [], right = [])
    r_reflector.data = dict(top=[], bottom=[], left = [], right = [])

def update_updates(input_string):
    update_messages.append(input_string)
    update_message = update_section_title
    for message in update_messages[-7:]:
        update_message+='\n'
        update_message+=message
    updates.text = str(update_message)

def update_image(data):
    new_data = dict()
    new_data['data'] = [data]
    display_data.data = new_data

#-----------------Toggling Between Different Choices-------------------#

def set_place_reflector():
    new_dict = {}
    new_dict['type'] = ["Reflector"]
    new_dict['shape'] = details.data['shape']
    details.data = new_dict

def set_place_source():
    new_dict = {}
    new_dict['type'] = ["Source"]
    new_dict['shape'] = details.data['shape']
    details.data = new_dict

def set_place_circle():
    new_dict = {}
    new_dict['type'] = details.data['type']
    new_dict['shape'] = ["Circle"]
    details.data = new_dict

def set_place_rectangle():
    new_dict = {}
    new_dict['type'] = details.data['type']
    new_dict['shape'] = ["Rectangle"]
    details.data = new_dict

def update_toggle_circle_rectangle():
    indicator = toggle_circle_rectangle.active
    if indicator == 0:
        set_place_circle()
    if indicator == 1:
        set_place_rectangle()

def update_toggle_source_reflector():
    indicator = toggle_source_reflector.active
    if indicator == 0:
        set_place_source()
    if indicator == 1:
        set_place_reflector()

#--------------------------Setting Variables----------------------------#

def set_Q_handler(attr, old, new):
    set_Q(new)

def set_L_handler(attr, old, new):
    set_L(new)


def set_Q(new):
    new_dict = {}
    new_dict['l'] = simulation_params.data['l']
    new_dict['Q'] = [new]
    simulation_params.data = new_dict

def set_L(new):
    new_dict = {}
    new_dict['Q'] = simulation_params.data['Q']
    new_dict['l'] = [new]
    simulation_params.data = new_dict

#-----------------------------------------------------------------------#

#----------------------------Simulating---------------------------------#
def run():
    simulating_model(objectList)

def simulating_model(in_objectList):
    #so far there are no checks to see if there is an error or intersection
    #source data
    c_a_r = query_object_list("Reflector", "Circle", in_objectList)
    c_a_s = query_object_list("Source", "Circle", in_objectList)
    r_a_r = query_object_list("Reflector","Rectangle",in_objectList)
    r_a_s = query_object_list("Source","Rectangle",in_objectList)

    sample = PM.RectangularSample(50,50)

    update_updates('Building Model')
    #placing reflectors
    for i in c_a_r:
        sample.placeCircularReflector(i.get_x_coord(),i.get_y_coord(),i.get_radius())

    for i in r_a_r:
        sample.placeRectangularReflector(i.get_x_coord(), i.get_y_coord(), i.get_width(), i.get_height())

    #placing sources
    for i in c_a_s:
        sample.placeCircularSource(i.get_x_coord(),i.get_y_coord(),i.get_radius())

    for i in r_a_s:
        sample.placeRectangularSource(i.get_x_coord(), i.get_y_coord(), i.get_width(), i.get_height())

    update_updates('Setting Parameters')
    #setting arbitrary omega and sigma values
    sigma = PM.S()
    omega = PM.O()
    sigma.set_sigma_values_RQ(float(simulation_params.data['l'][0]),float(simulation_params.data['Q'][0]))
    omega.set_omega_values(1,1)

    update_updates('Running Simulation')
    #running the simulation
    sample.run(omega,sigma,density = 200)

    update_updates('Done!')
    results = sample.cast_solution_to_Array()

    display_data.data = dict(data = [results[0]])
#-----------------------------------------------------------------------#
#-----------------------------Text Input--------------------------------#

quality_factor_input = TextInput(value="100", title="Quality Factor")
quality_factor_input.on_change("value",set_Q_handler)

plasmon_wavelength_input = TextInput(value="4", title="Plasmon Wavelength")
plasmon_wavelength_input.on_change("value",set_L_handler)

#-----------------------------------------------------------------------#

#--------------------------------Buttons--------------------------------#


# add a button widget and configure with the call back
button1 = Button(label="Run Simulation")
button1.on_click(run)

#add a button widget to reset the board
button2 = Button(label="Reset Board")
button2.on_click(reset)

#add a button to switch between source and reflector
button3 = Button(label="Toggle Reflector")
button3.on_click(set_place_reflector)

button4 = Button(label="Toggle Source")
button4.on_click(set_place_source)

button5 = Button(label="Toggle Circle")
button5.on_click(set_place_circle)

button6 = Button(label="Toggle Rectangle")
button6.on_click(set_place_rectangle)

#Button 7
# button7 = Button(label="Undo")
# button7.on_click(undo_place)

toggle_source_reflector = RadioButtonGroup(labels=["Source", "Reflector"], active=0)
toggle_source_reflector.on_change('active', lambda attr ,old, new: update_toggle_source_reflector())

toggle_circle_rectangle = RadioButtonGroup(labels=["Circle", "Rectangle"], active=0)
toggle_circle_rectangle.on_change('active', lambda attr ,old, new: update_toggle_circle_rectangle())

#-----------------------------------------------------------------------#

clickable_display.on_event(DoubleTap, callback)

#toggle switches
# toggle_switches = row(toggle_source_reflector,toggle_circle_rectangle)

# toggle_source_reflector = row(button3, button4)
# toggle_circle_rectangle = row(button5, button6)


button_display = column(button1, button2, toggle_source_reflector,toggle_circle_rectangle, width = 200)
circular_slider_display = column(circular_title,circular_radius)
rectangular_slider_display = column(rectangular_title, rectangular_width, rectangular_height)
rotation_display = column(rotation_title,rotation)
simulation_params_display = column(simulation_params_title,quality_factor_input,plasmon_wavelength_input)

# grid=gridplot([[button_display,rotation_display],[circular_slider_display,rectangular_slider_display]])

right_display = column(button_display,circular_slider_display,rectangular_slider_display,rotation_display,simulation_params_display,updates)
curdoc().add_root(row(clickable_display,output_display,right_display))
