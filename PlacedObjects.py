#--------------Object oriented part------------#
#Going to need to be able to control lists of these, transform those lists into four CDSs for display

class PlacedObject(object):

    def __init__(self,x_coord, y_coord):
        self.x_coord = x_coord
        self.y_coord = y_coord

    def get_x_coord(self):
        return self.x_coord

    def get_y_coord(self):
        return self.y_coord

    def set_reflector(self):
        self.source = False

    def set_source(self):
        self.source = True

    def is_reflector(self):
        return not self.source

    def is_source(self):
        return self.source

    def get_shape(self):
        return None

class RectangularObject(PlacedObject):

    def __init__(self,x_coord, y_coord, width, height):
        super(RectangularObject, self).__init__(x_coord, y_coord)
        self.width = width
        self.height = height
        self.shape = "Rectangle"

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_shape(self):
        return self.shape


class CircularObject(PlacedObject):

    def __init__(self,x_coord, y_coord, radius):
        super(CircularObject, self).__init__(x_coord, y_coord)
        self.radius = radius
        self.shape = "Circle"

    def get_radius(self):
        return self.radius

class PolygonalObject(PlacedObject):

    def __init__(self):
        pass

    def get_shape(self):
        return self.shape


# General Functionality Test
# test1 = PlacedObject(1,2)
# test2 = RectangularObject(1,2,3,4)
# test3 = CircularObject(1,2,3)
#
# print "General Object"
# print "x_coord: {}, y_coord: {}".format(test1.get_x_coord(),test1.get_y_coord())
#
# print "Rectangle Object"
# print "x_coord: {}, y_coord: {}, width: {}, height: {}".format(test2.get_x_coord(),test2.get_y_coord(),test2.get_width(),test2.get_height())
#
# print "Circular Object"
# print "x_coord: {}, y_coord: {}, radius: {}".format(test3.get_x_coord(),test3.get_y_coord(), test3.get_radius())
