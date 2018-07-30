from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Column
from bokeh.io import curdoc
from bokeh.events import DoubleTap
from bokeh.models import Button
from bokeh.layouts import column, row
from bokeh.models.widgets import PreText
import numpy as np

bound = 50
coordList=[]
display_data= ColumnDataSource({'data' : np.zeros((50,50))})

TOOLS = "tap,reset"

p = figure(title='Double click to leave a dot.',
           tools=TOOLS,width=700,height=700,
           x_range=(-bound, bound), y_range=(-bound, bound))

d = figure(title='Display', plot_width=350, plot_height=350,
              tools='pan,wheel_zoom,box_select,reset')

d.image('data', source=display_data,
         x=0, y=0, dw=bound, dh=bound,
         palette="Viridis256")

stats = PreText(text='', width=350)

source = ColumnDataSource(data=dict(x=[], y=[]))
p.circle(source=source,x='x',y='y')

#add a dot where the click happened
def callback(event):
    Coords=(event.x,event.y)
    coordList.append(Coords)
    update_data = dict(x=[i[0] for i in coordList], y=[i[1] for i in coordList])
    source.data = update_data
    update_stats(update_data)

def displayPrinted():
    print coordList
    print display_data

def update_stats(data):
    stats.text = str(data)
    update_image

def update_image(data):
    print "Should Work"
    new_data = dict()
    new_data['data'] = image_data[randint(0, num_images - 1)]
    display_data = new_data


# add a button widget and configure with the call back
button = Button(label="Print Out Data")
button.on_click(displayPrinted)


p.on_event(DoubleTap, callback)
layout=Column(p)

widgets_display = column(button,d,stats)
curdoc().add_root(row(layout,widgets_display))
