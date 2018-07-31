from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Column
from bokeh.io import curdoc
from bokeh.events import DoubleTap
from bokeh.models import Button, ContinuousColorMapper, BasicTicker, ColorBar
from bokeh.models.widgets import Slider
from bokeh.layouts import column, row
from bokeh.models.widgets import PreText
import Plasmon_Modeling as PM
import numpy as np

#setting up data
bound = 50
coordList=[]
sourceList = []
source_radius_list = []
reflector_radius_list = []
display_data= ColumnDataSource({'data' : [np.zeros((50,50))]})
is_source = ColumnDataSource({'source' : [True], 'shape': "Circle"})

update_section_title = "Update Section"
update_messages = []

#setup plotting
p = figure(title='Double click to leave a dot.',
           tools="tap,reset",width=700,height=700,
           x_range=(0, bound), y_range=(0, bound))

#Currently having trouble implementing the ColorBar

# color_mapper = ContinuousColorMapper(palette="Viridis256", low=1, high=10)

d = figure(title='Display', plot_width=700, plot_height=700,
           x_range=(0, bound), y_range=(0, bound), tools='pan,wheel_zoom,box_select,reset')

d.image('data', source=display_data, palette="Viridis256",
         x=0, y=0, dw=bound, dh=bound)

# color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
#                      label_standoff=12, border_line_color=None, location=(0,0))
#
# d.add_layout(color_bar, 'right')

#text items
stats = PreText(text='Update Section', width=350)
circular_title = PreText(text='Circular Reflectors/Sources', width=350)
rectangular_title =  PreText(text='Rectangular Reflectors/Sources', width=350)

#setting up sliders
circular_radius = Slider(title = "Radius",value = 2, start = 1, end = 10, step = 1)
rectangular_width = Slider(title = "Width",value = 2, start = 1, end = 10, step = 1)
rectangular_height = Slider(title = "Height",value = 2, start = 1, end = 10, step = 1)

#setting up the source and reflector indicators
source = ColumnDataSource(data=dict(x=[], y=[], r = []))
p.circle(source=source,x='x',y='y',radius = 'r', color="navy", alpha = 0.5)

reflector = ColumnDataSource(data=dict(x=[], y=[], r = []))
p.circle(source=reflector,x='x',y='y',radius = 'r', color = "red", alpha = 0.5)

#add a dot where the click happened
def callback(event):
    Coords=(event.x,event.y)
    coordList.append(Coords)
    source_or_reflector = is_source.data['Bool'][0]
    sourceList.append(source_or_reflector)

    print coordList
    print sourceList

    if source_or_reflector:
        #modifying the source radius list and updating the data
        source_radius_list.append(circular_radius.value)
        update_data = dict(x=[coordList[i][0] for i in range(len(coordList)) if sourceList[i]], y=[coordList[i][1] for i in range(len(coordList)) if sourceList[i]], r =source_radius_list)
        source.data = update_data
    else:
        #modifying the reflector radius list and updating the data
        reflector_radius_list.append(circular_radius.value)
        update_data = dict(x=[coordList[i][0] for i in range(len(coordList)) if not sourceList[i]], y=[coordList[i][1] for i in range(len(coordList)) if not sourceList[i]], r = reflector_radius_list)
        reflector.data = update_data

def run():
    simulating_model()

def simulating_model():
    #so far there are no checks to see if there is an error or intersection

    #source data
    source_x = source.data['x']
    source_y = source.data['y']
    source_r = source.data['r']

    #reflector data
    reflector_x = reflector.data['x']
    reflector_y = reflector.data['y']
    reflector_r = reflector.data['r']

    sample = PM.RectangularSample(50,50)

    update_stats('Building Model')
    #placing reflectors
    for i in range(len(reflector_x)):
        sample.placeCircularReflector(reflector_x[i],reflector_y[i],reflector_r[i])

    #placing sources
    for i in range(len(source_x)):
        sample.placeCircularSource(source_x[i],source_y[i],source_r[i])

    update_stats('Setting Parameters')
    #setting arbitrary omega and sigma values
    sigma = PM.S()
    omega = PM.O()
    sigma.set_sigma_values(1,10)
    omega.set_omega_values(1,1)

    update_stats('Running Simulation')
    #running the simulation
    sample.run(omega,sigma,density = 200)

    update_stats('Done!')
    awa = sample.cast_solution_to_AWA()

    real_part = awa[0].T
    im_part = awa[1].T

    display_data.data = dict(data = [real_part])

def reset():

    del coordList[:]
    del sourceList[:]
    del source_radius_list[:]
    del reflector_radius_list[:]
    del update_messages[:]

    display_data.data = dict(data = [np.random.randn(50,50)])
    source.data = dict(x=[], y=[], r = [])
    reflector.data = dict(x=[], y=[], r = [])




def update_stats(input_string):
    update_messages.append(input_string)
    update_message = update_section_title
    for message in update_messages[-7:]:
        update_message+='\n'
        update_message+=message
    stats.text = str(update_message)

def update_image(data):
    new_data = dict()
    new_data['data'] = [data]
    display_data.data = new_data

def set_place_reflector():
    new_dict = {}
    new_dict['Bool'] = [False]
    is_source.data = new_dict

def set_place_source():
    new_dict = {}
    new_dict['Bool'] = [True]
    is_source.data = new_dict

def set_place_circle():

def set_place_rectangle():



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

p.on_event(DoubleTap, callback)
layout=Column(p)

toggle_source_reflector = row(button3,button4)

button_display = column(button1, button2, button3, button4, width = 50)
circular_slider_display = column(circular_title,circular_radius)
rectangular_slider_display = column(rectangular_title, rectangular_width, rectangular_height)
right_display = column(button_display,circular_slider_display,rectangular_slider_display,stats)
curdoc().add_root(row(layout,d,right_display))
