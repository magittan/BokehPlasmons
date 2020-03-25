from bokeh.io import curdoc
from bokeh.events import DoubleTap
from bokeh.layouts import column, row, widgetbox, gridplot
from bokeh.models import CustomJS,ColumnDataSource, Column, Button, ContinuousColorMapper, BasicTicker, ColorBar, OpenURL
from bokeh.models.widgets import PreText, RadioButtonGroup, TextInput, Slider, Div, Dropdown
from bokeh.plotting import figure, show
import Plasmon_Modeling as PM
import PlacedObjects as PO
import numpy as np

# Add sources and reflectors to certain clicked points
def DoubleClickCallback(event):
    coordinates = (event.x,event.y)
    type = details.data['type'][0]
    shape = details.data['shape'][0]

    #print(coordinates)
    #print(type)
    #print(shape)

    #Checking the shape
    if shape == 'Rectangle':
        angle = rotation.value
        width = rectangular_width.value
        height = rectangular_height.value
        input_object = PO.RectangularObject(coordinates[0]-width/2,coordinates[1]-height/2, width, height, angle)
        input_object.set_type(type)
        AddRectangle(input_object, objectList)

    elif shape == 'Circle':
        input_object = PO.CircularObject(coordinates[0],coordinates[1],circular_radius.value)
        input_object.set_type(type)
        AddCircle(input_object, objectList)

def AddRectangle(input_object, in_objectList):
    #print('Adding rectangle')
    search_type = input_object.get_type()
    in_objectList.append(input_object)

    temp = QueryObjectList(search_type,'Rectangle',in_objectList)
    xs = [[j[0] for j in i.get_coordinates()] for i in temp]
    ys = [[j[1] for j in i.get_coordinates()] for i in temp]

    update_data = dict(x = xs, y=ys)

    if search_type == 'Source':
        p_source.data = update_data
    elif search_type == 'Reflector':
        p_reflector.data = update_data

def AddCircle(input_object, in_objectList):
    search_type = input_object.get_type()
    in_objectList.append(input_object)

    temp = QueryObjectList(search_type,'Circle',in_objectList)
    x = np.array([i.get_x_coord() for i in temp])
    y = np.array([i.get_y_coord() for i in temp])
    r = np.array([i.get_radius() for i in temp])

    update_data = dict(x=x ,y=y ,r=r)

    if search_type == 'Source':
        c_source.data = update_data
    elif search_type == 'Reflector':
        c_reflector.data = update_data

def UndoPlace(event):
    search_type = objectList[-1].get_type()
    search_shape = objectList[-1].get_shape()
    del objectList[-1]
    #objectList = objectList[:-1]

    if search_shape == 'Rectangle':
        temp = QueryObjectList(search_type,'Rectangle',objectList)
        xs = [[j[0] for j in i.get_coordinates()] for i in temp]
        ys = [[j[1] for j in i.get_coordinates()] for i in temp]

        update_data = dict(x = xs, y=ys)

        if search_type == 'Source':
            p_source.data = update_data
        elif search_type == 'Reflector':
            p_reflector.data = update_data

    elif search_shape == 'Circle':
        temp = QueryObjectList(search_type,'Circle',objectList)
        x = np.array([i.get_x_coord() for i in temp])
        y = np.array([i.get_y_coord() for i in temp])
        r = np.array([i.get_radius() for i in temp])

        update_data = dict(x=x ,y=y ,r=r)

        if search_type == 'Source':
            c_source.data = update_data
        elif search_type == 'Reflector':
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

#-----------------Toggling Between Different Choices-------------------#

def SetPlaceReflector():
    new_dict = {}
    new_dict['type'] = ['Reflector']
    new_dict['shape'] = details.data['shape']
    details.data = new_dict

def SetPlaceSource():
    new_dict = {}
    new_dict['type'] = ['Source']
    new_dict['shape'] = details.data['shape']
    details.data = new_dict

def SetPlaceCircle():
    new_dict = {}
    new_dict['type'] = details.data['type']
    new_dict['shape'] = ['Circle']
    details.data = new_dict

def SetPlaceRectangle():
    new_dict = {}
    new_dict['type'] = details.data['type']
    new_dict['shape'] = ['Rectangle']
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

def UpdatePlotValue(attr,old,new):
    plot_value_dropdown.label = new
    w = int(mesh_width_input.value)
    h = int(mesh_height_input.value)
    output_display.image(new, source = display_data.data,x=0,y=0,dw=w,dh=h,palette='Viridis256')

def TestURLCallback(event):
    with open('./myapp/static/test1.dat','w+') as f:
        f.write('test\ntest test\n')
    url = 'localhost:5006/myapp/static/test1.dat'


#--Setting Variables----#

def SetMeshWidth(attr,old,new):
    x = int(new)
    y = int(mesh_height_input.value)
    display_data.data['data'] = [np.zeros((x,y))]
    clickable_display.x_range.end = x
    clickable_display.y_range.end = y
    output_display.x_range.end = x
    output_display.y_range.end = y
    output_display.image('data', source=display_data, palette='Viridis256',x=0, y=0, dw=x_bound, dh=y_bound)

def SetMeshHeight(attr,old,new):
    x = int(mesh_width_input.value)
    y = int(new)
    display_data.data['data'] = [np.zeros((x,y))]
    clickable_display.x_range.end = x
    clickable_display.y_range.end = y
    output_display.x_range.end = x
    output_display.y_range.end = y
    output_display.image('data', source=display_data, palette='Viridis256',x=0, y=0, dw=x, dh=y)

def SetQ(attr, old, new):
    new_dict = {}
    new_dict['l'] = simulation_params.data['l']
    new_dict['Q'] = [new]
    simulation_params.data = new_dict

def SetL(attr, old, new):
    new_dict = {}
    new_dict['Q'] = simulation_params.data['Q']
    new_dict['l'] = [new]
    simulation_params.data = new_dict

#----Simulating---------#

def Run():
    SimulateModel(objectList)

def Reset():
    del objectList[:]
    del update_messages[:]

    w,h = int(mesh_width_input.value),int(mesh_height_input.value)
    display_data.data = dict(data = [np.zeros((w,h))])
    c_source.data = dict(x=[], y=[], r = [])
    c_reflector.data = dict(x=[], y=[], r = [])
    p_source.data = dict(x=[], y=[])
    p_reflector.data = dict(x=[], y=[])

    clickable_display.circle(source=c_source,x='x',y='y',radius = 'r', color='navy', alpha = 0.5)
    clickable_display.circle(source=c_reflector,x='x',y='y',radius = 'r', color = 'red', alpha = 0.5)
    clickable_display.patches(source = p_source,xs='x',ys='y',color ='navy', alpha = 0.5)
    clickable_display.patches(source = p_reflector,xs='x',ys='y',color ='red', alpha = 0.5)

def SimulateModel(in_objectList):
    #so far there are no checks to see if there is an error or intersection
    #source data
    c_a_r = QueryObjectList('Reflector', 'Circle', in_objectList)
    c_a_s = QueryObjectList('Source', 'Circle', in_objectList)
    r_a_r = QueryObjectList('Reflector','Rectangle',in_objectList)
    r_a_s = QueryObjectList('Source','Rectangle',in_objectList)

    w = int(mesh_width_input.value)
    h = int(mesh_height_input.value)
    sample = PM.RectangularSample(w,h)

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
    lam = float(lambda_input.value)
    phi = float(phi_input.value)*np.pi/180

    UpdateUpdates('Running Simulation')
    #running the simulation
    sample.run(omega,sigma,density = int(mesh_density_input.value), _lam=lam,_phi=phi)

    UpdateUpdates('Done!')
    results = sample.cast_solution_to_Array()

    real = results[0]
    imag = results[1]
    mag = np.sqrt(real**2 + imag**2)
    phi = np.arctan2(imag,real)

    display_data.data['Real Part'] = [real]
    display_data.data['Imaginary Part'] = [imag]
    display_data.data['Magnitude'] = [mag]
    display_data.data['Phase'] = [phi]
    w = int(mesh_width_input.value)
    h = int(mesh_height_input.value)
    print(plot_value_dropdown.value)
    new_d = ColumnDataSource({'data': display_data.data[plot_value_dropdown.value]})
    output_display.image('data',source=new_d,x=0,y=0,dw=w,dh=h, palette='Viridis256')
    #output_display.image(plot_value_dropdown.value,source=display_data,x=0,y=0,dw=w,dh=h, palette='Viridis256')

def GenerateMesh():
    _GenerateMesh(objectList)

def _GenerateMesh(ol):
    w = int(mesh_width_input.value)
    h = int(mesh_height_input.value)
    r_a_r = QueryObjectList('Reflector','Rectangle',ol)
    c_a_r = QueryObjectList('Reflector', 'Circle', ol)
    mesh_sample = PM.RectangularSample(w,h)
    for i in c_a_r:
        mesh_sample.placeCircularReflector(i.get_x_coord(),i.get_y_coord(),i.get_radius())

    for i in r_a_r:
        mesh_sample.placeRectangularReflector(i.get_x_coord(), i.get_y_coord(), i.get_width(), i.get_height(), i.get_angle())

    density = int(mesh_density_input.value)
    mesh_sample.getMesh(density = density, to_plot=True)

# Data Sources
x_bound, y_bound = 50,50
objectList = []
display_data= ColumnDataSource({
        'Real Part': [np.zeros((x_bound,y_bound))],
        'Imaginary Part': [np.zeros((x_bound,y_bound))],
        'Magnitude': [np.zeros((x_bound,y_bound))],
        'Phase': [np.zeros((x_bound,y_bound))]
})
details = ColumnDataSource({'type' : ['Source'], 'shape': ['Rectangle'],'rotation': [0]})
simulation_params = ColumnDataSource({'Q' : [100], 'l': [4]})
c_source = ColumnDataSource(data=dict(x=[], y=[], r = []))
c_reflector = ColumnDataSource(data=dict(x=[], y=[], r = []))
p_source = ColumnDataSource(data=dict(x=[],y=[]))
p_reflector = ColumnDataSource(data=dict(x=[],y=[]))

# Update message section
update_section_title = 'Update Section'
update_messages = []

# Displays
clickable_display = figure(title='Double click to place source/reflector.',
           tools='tap,reset',
           x_range=(0, x_bound), y_range=(0, y_bound),
           x_axis_label = "x [mesh units]", y_axis_label = "y [mesh units]")
output_display = figure(title='Sample Response Display',
            tools='wheel_zoom,box_select,save,reset',
            x_range=(0, x_bound), y_range=(0, y_bound),
            x_axis_label = "x [mesh units]", y_axis_label = "y [mesh units]")
clickable_display.xaxis.axis_label_text_font_style = "normal"
clickable_display.yaxis.axis_label_text_font_style = "normal"
output_display.xaxis.axis_label_text_font_style = "normal"
output_display.yaxis.axis_label_text_font_style = "normal"

# Labels
updates_pretext = PreText(text='Update Section', width=300)
#circular_pretext = PreText(text='Circular Reflectors/Sources', width=300)
#rectangular_pretext =  PreText(text='Rectangular Reflectors/Sources', width=300)
circular_pretext =  Div(text='Circular Reflectors/Sources', style={"font-family": "Arial", "font-size": "15px"}, width=300)
rectangular_pretext =  Div(text='Rectangular Reflectors/Sources', style={"font-family": "Arial", "font-size": "15px"}, width=300)
#rotation_pretext =  PreText(text='Rotation (Degrees)', width=300)
#simulation_params_pretext =  PreText(text='Rotation (Degrees)', width=300)

# Sliders
circular_radius = Slider(title = 'Radius',value = 2, start = 1, end = 10, step = 1,width=200)
rectangular_width = Slider(title = 'Width',value = 8, start = 1, end = 10, step = 1,width=200)
rectangular_height = Slider(title = 'Height',value = 2, start = 1, end = 10, step = 1,width=200)
rotation = Slider(title = 'Rotation (Degrees)',value = 0, start = 0, end = 360, step = 15,width=200)

# Text Input
mesh_width_input = TextInput(value='50', title='Mesh Width', width=100)
mesh_height_input = TextInput(value='50', title='Mesh Height', width=100)
quality_factor_input = TextInput(value='100', title='Plasmon Q', width=100)
plasmon_wavelength_input = TextInput(value='4', title='Plasmon '+u"\u03BB", width=100)
mesh_density_input = TextInput(value='100', title='Mesh Density',width=200)
lambda_input = TextInput(value='100', title='Excitation '+u"\u03BB",width=100)
phi_input = TextInput(value='90', title='Excitation '+u"\u03B8",width=100)

# Buttons
undo_button = Button(label='Undo Last Placement', width = 200)
run_button = Button(label='Run Simulation',width=150)
reset_button = Button(label='Reset Board',width=150)
generate_mesh_button = Button(label='Generate Mesh',width=200)

jscallback = CustomJS(code='window.open("/myapp/static/test1.dat");')
test_url_button = Button(label='Test URL Generation',width=200,callback=jscallback)

# Radio buttons
toggle_source_reflector = RadioButtonGroup(labels=['Source', 'Reflector'], active=0,width=200)
toggle_circle_rectangle = RadioButtonGroup(labels=['Circle', 'Rectangle'], active=1,width=200)

# Dropdown
plot_value_dropdown = Dropdown(label='Real Part',value='Real Part',menu=[
    'Real Part',
    'Imaginary Part',
    'Phase',
    'Magnitude'
])

# Callbacks
run_button.on_click(Run)
reset_button.on_click(Reset)
generate_mesh_button.on_click(GenerateMesh)
undo_button.on_click(UndoPlace)
#test_url_button.on_click(TestURLCallback)
toggle_source_reflector.on_change('active', lambda attr ,old, new: UpdateTypeToggle())
toggle_circle_rectangle.on_change('active', lambda attr ,old, new: UpdateShapeToggle())
quality_factor_input.on_change('value',SetQ)
plasmon_wavelength_input.on_change('value',SetL)
mesh_width_input.on_change('value', SetMeshWidth)
mesh_height_input.on_change('value', SetMeshHeight)
plot_value_dropdown.on_change('value', UpdatePlotValue)
clickable_display.on_event(DoubleTap, DoubleClickCallback)

clickable_display.circle(source=c_source,x='x',y='y',radius = 'r', color='navy', alpha = 0.5)
clickable_display.circle(source=c_reflector,x='x',y='y',radius = 'r', color = 'red', alpha = 0.5)
clickable_display.patches(source = p_source,xs='x',ys='y',color ='navy', alpha = 0.5)
clickable_display.patches(source = p_reflector,xs='x',ys='y',color ='red', alpha = 0.5)
output_display.image('Real Part', source=display_data,x=0, y=0, dw=x_bound, dh=y_bound, palette='Viridis256')

# Set up main app
spacer = column(width = 50)
mesh_params_display = row(spacer,column(row(mesh_width_input,mesh_height_input), mesh_density_input, generate_mesh_button),spacer)
toggle_button_display = row(spacer,column(undo_button, toggle_source_reflector,toggle_circle_rectangle),spacer, width = 300)
circular_slider_display = column(circular_pretext,circular_radius)
rectangular_slider_display = column(rectangular_pretext, rectangular_width, rectangular_height)
rotation_display = column(rotation)
simulation_params_display = row(spacer,quality_factor_input,plasmon_wavelength_input,spacer)
dirichlet_params_display = row(spacer,lambda_input, phi_input,spacer)
button_display = row(reset_button,run_button)
right_display_column_elements = [
    mesh_params_display,
    Div(width = 300, height = 1, background = '#000000'),
    toggle_button_display,
    Div(width = 300, height = 1, background = '#000000'),
    circular_slider_display,
    Div(width = 300, height = 1, background = '#000000'),
    rectangular_slider_display,
    rotation_display,
    Div(width = 300, height = 1, background = '#000000'),
    simulation_params_display,
    dirichlet_params_display,
    plot_value_dropdown,
    button_display,
    #test_url_button,
    Div(width = 300, height = 1, background = '#000000'),
    updates_pretext
]

right_display = column(right_display_column_elements)
curdoc().add_root(row(row(clickable_display,output_display,sizing_mode='scale_both'),row(spacer,right_display,spacer)))
