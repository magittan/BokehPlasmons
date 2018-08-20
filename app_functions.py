import numpy as np
#This class interacts with PlacedObjects.py
import PlacedObjects as PO

def query_object_list(in_type, in_shape, objectList):
    return [i for i in objectList if (i.get_type()==in_type) and (i.get_shape()==in_shape)]

#expect the input to be of the form [[x0,y0],[x1,y1],[x2,y2],[x3,y3],[x4,y4],...]
def rotate_object(vertices_list, rotation_degrees):
    output_vertices = []
    theta = np.radians(rotation_degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    for vertex in vertices_list:
        output_vertices.append([vertex[0]*c-vertex[1]*s,vertex[0]*c+vertex[1]*s])
    return output_vertices
        #Complete this and then run the bokeh webapp and finished convertig from 2 to 3, running into problems with h5py

#Test class
test1= PO.RectangularObject(0,1,2,3)
test1.set_type("Source")
test2= PO.RectangularObject(1,1,2,3)
test2.set_type("Source")
test3= PO.RectangularObject(2,1,2,3)
test3.set_type("Source")
test4= PO.RectangularObject(3,1,2,3)
test4.set_type("Reflector")
test5= PO.RectangularObject(4,1,2,3)
test5.set_type("Reflector")

test6 = PO.CircularObject(0,1,2)
test6.set_type("Source")
test7 = PO.CircularObject(1,1,2)
test7.set_type("Source")
test8 = PO.CircularObject(2,1,2)
test8.set_type("Source")
test9 = PO.CircularObject(3,1,2)
test9.set_type("Reflector")
test0 = PO.CircularObject(4,1,2)
test0.set_type("Reflector")

objectList = [test0 ,test1 ,test2 ,test3 ,test4 ,test5 ,test6 ,test7 ,test8 ,test9]

#Rectangular and Source
r_a_s=query_object_list("Source","Rectangle",objectList)

#Rectangular and Reflector
r_a_r=query_object_list("Reflector","Rectangle",objectList)

#Circular and Source
c_a_s=query_object_list("Source","Circle",objectList)

#Circular and Reflector
c_a_r=query_object_list("Reflector","Circle",objectList)

print "Rectangular and Sources"
print "-------------------------------"
for i in r_a_s:
    print "Shape: {}, Type: {}, X-coord: {}. Y-coord: {}".format(i.get_shape(),i.get_type() \
                                                                ,i.get_x_coord(),i.get_y_coord())
print "\n"
print "Rectangular and Reflectors"
print "-------------------------------"
for i in r_a_r:
    print "Shape: {}, Type: {}, X-coord: {}. Y-coord: {}".format(i.get_shape(),i.get_type() \
                                                                ,i.get_x_coord(),i.get_y_coord())
print "\n"
print "Circular and Sources"
print "-------------------------------"
for i in c_a_s:
    print "Shape: {}, Type: {}, X-coord: {}. Y-coord: {}".format(i.get_shape(),i.get_type() \
                                                                ,i.get_x_coord(),i.get_y_coord())
print "\n"
print "Circular and Reflectors"
print "-------------------------------"
for i in c_a_r:
    print "Shape: {}, Type: {}, X-coord: {}. Y-coord: {}".format(i.get_shape(),i.get_type() \
                                                                ,i.get_x_coord(),i.get_y_coord())
print "\n"
