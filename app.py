from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Column
from bokeh.io import curdoc
from bokeh.events import DoubleTap
from bokeh.models import Button, ContinuousColorMapper, BasicTicker, ColorBar
from bokeh.models.widgets import Slider
from bokeh.layouts import column, row, widgetbox
from bokeh.models.widgets import PreText, RadioButtonGroup
import Plasmon_Modeling as PM
import PlacedObjects as PO
import numpy as np

#setting up data
bound = 50
objectList= []
display_data= ColumnDataSource({'data' : [np.zeros((50,50))]})
details = ColumnDataSource({'type' : ["Source"], 'shape': ["Circle"]})

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

#setting up sliders
circular_radius = Slider(title = "Radius",value = 2, start = 1, end = 10, step = 1)
rectangular_width = Slider(title = "Width",value = 2, start = 1, end = 10, step = 1)
rectangular_height = Slider(title = "Height",value = 2, start = 1, end = 10, step = 1)

#setting up the source and reflector indicators for circles
c_source = ColumnDataSource(data=dict(x=[], y=[], r = []))
clickable_display.circle(source=c_source,x='x',y='y',radius = 'r', color="navy", alpha = 0.5)

c_reflector = ColumnDataSource(data=dict(x=[], y=[], r = []))
clickable_display.circle(source=c_reflector,x='x',y='y',radius = 'r', color = "red", alpha = 0.5)

#setting up the source and reflector indicators for rectangles
r_source = ColumnDataSource(data=dict(top=[], bottom=[], left = [], right = []))
clickable_display.quad(source=r_source,top='top',bottom='bottom',left = 'left', right = "right", color="navy", alpha = 0.5)

r_reflector = ColumnDataSource(data=dict(top=[], bottom=[], left = [], right = []))
clickable_display.quad(source=r_reflector,top='top',bottom='bottom',left = 'left', right = "right", color = "red", alpha = 0.5)

#Add sources and reflectors to certain clicked points
def callback(event):
    coordinates = (event.x,event.y)
    type = details.data['type'][0]
    shape = details.data['shape'][0]

    print coordinates
    print type
    print shape

    #Checking the shape
    if shape == "Rectangle":
        input_object = PO.RectangularObject(coordinates[0],coordinates[1],rectangular_width.value, rectangular_height.value)
        input_object.set_type(type)
        add_rectangle(input_object, objectList)

    elif shape == "Circle":
        input_object = PO.CircularObject(coordinates[0],coordinates[1],circular_radius.value)
        input_object.set_type(type)
        add_circle(input_object, objectList)


def add_rectangle(input_object, in_objectList):
    search_type = input_object.get_type()
    in_objectList.append(input_object)

    temp = query_object_list(search_type,"Rectangle",in_objectList)
    x = np.array([i.get_x_coord() for i in temp])
    y = np.array([i.get_y_coord() for i in temp])
    w = np.array([i.get_width() for i in temp])
    h = np.array([i.get_height() for i in temp])

    top = y+h
    bottom = y
    left = x
    right = x+w

    update_data = dict(top = top, bottom = bottom, left = left, right = right)

    if search_type == "Source":
        r_source.data = update_data
    elif search_type == "Reflector":
        r_reflector.data = update_data

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
        x = np.array([i.get_x_coord() for i in temp])
        y = np.array([i.get_y_coord() for i in temp])
        w = np.array([i.get_width() for i in temp])
        h = np.array([i.get_height() for i in temp])

        top = y+h
        bottom = y
        left = x
        right = x+w

        update_data = dict(top = top, bottom = bottom, left = left, right = right)

        if search_type == "Source":
            r_source.data = update_data
        elif search_type == "Reflector":
            r_reflector.data = update_data

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
    sigma.set_sigma_values(1,10)
    omega.set_omega_values(1,1)

    update_updates('Running Simulation')
    #running the simulation
    sample.run(omega,sigma,density = 200)

    update_updates('Done!')
    awa = sample.cast_solution_to_AWA()

    real_part = awa[0].T
    im_part = awa[1].T

    display_data.data = dict(data = [real_part])
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

def update():
    print toggle_circle_rectangle.active

# toggle_circle_rectangle = RadioButtonGroup(
#         labels=["Rectangle", "Circle"], active=0)
# toggle_circle_rectangle.on_change('active', lambda attr ,old, new, update())


#-----------------------------------------------------------------------#

clickable_display.on_event(DoubleTap, callback)

toggle_source_reflector = row(button3, button4)
toggle_circle_rectangle = row(button5, button6)

button_display = column(button1, button2, toggle_source_reflector,toggle_circle_rectangle, width = 50)
circular_slider_display = column(circular_title,circular_radius)
rectangular_slider_display = column(rectangular_title, rectangular_width, rectangular_height)
right_display = column(button_display,circular_slider_display,rectangular_slider_display,updates)
curdoc().add_root(row(clickable_display,output_display,right_display))
