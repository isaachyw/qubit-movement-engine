import math
import random
from ast import literal_eval
import re
import matplotlib.pyplot as plt
# A small tolerance for comparing floats for equality
TOL = 1.e-5
# psi = 1/phi where phi is the Golden ratio, sqrt(5)+1)/2
psi = (math.sqrt(5) - 1) / 2
# psi**2 = 1 - psi
psi2 = 1 - psi

class RobinsonTriangle:
    """
    A class representing a Robinson triangle and the rhombus formed from it.

    """

    def __init__(self, A, B, C):
        """
        Initialize the triangle with the ordered vertices. A and C are the
        vertices at the equal base angles; B is at the vertex angle.

        """
 
        self.A, self.B, self.C = A, B, C

    def centre(self):
        """
        Return the position of the centre of the rhombus formed from two
        triangles joined by their bases.

        """

        return (self.A + self.C) / 2

    def path(self, rhombus=True):
        """
        Return the SVG "d" path element specifier for the rhombus formed
        by this triangle and its mirror image joined along their bases. If
        rhombus=False, the path for the triangle itself is returned instead.

        """

        AB, BC = self.B - self.A, self.C - self.B 
        xy = lambda v: (v.real, v.imag)
        if rhombus:
            return 'm{},{} l{},{} l{},{} l{},{}z'.format(*xy(self.A) + xy(AB)
                                                        + xy(BC) + xy(-AB))
        return 'm{},{} l{},{} l{},{}z'.format(*xy(self.A) + xy(AB)
                                                        + xy(BC))
    # I just want the points, so add a new function
    def lufter(self, rhombus=True):
        """
        Return the SVG "d" path element specifier for the rhombus formed
        by this triangle and its mirror image joined along their bases. If
        rhombus=False, the path for the triangle itself is returned instead.

        """

        AB, BC = self.B - self.A, self.C - self.B 
        xy = lambda v: (v.real, v.imag)
        if rhombus:
            ()
            return '({},{}) ({},{}) ({},{}) ({},{})'.format(*xy(self.A) + xy(AB)
                                                        + xy(BC) + xy(-AB))
        return '({},{}) ({},{}) ({},{})'.format(*xy(self.A) + xy(AB)
                                                        + xy(BC))


    def get_arc_d(self, U, V, W, half_arc=False):
        """
        Return the SVG "d" path element specifier for the circular arc between
        sides UV and UW, joined at half-distance along these sides. If
        half_arc is True, the arc is at the vertex of a rhombus; if half_arc
        is False, the arc is drawn for the corresponding vertices of a
        Robinson triangle.

        """

        start = (U + V) / 2
        end = (U + W) / 2
        # arc radius
        r = abs((V - U) / 2)

        if half_arc:
            # Find the endpoint of the "half-arc" terminating on the triangle
            # base
            UN = V + W - 2*U
            end = U + r * UN / abs(UN)

        # ensure we draw the arc for the angular component < 180 deg
        cross = lambda u, v: u.real*v.imag - u.imag*v.real
        US, UE = start - U, end - U
        if cross(US, UE) > 0:
            start, end = end, start
        return 'M {} {} A {} {} 0 0 0 {} {}'.format(start.real, start.imag,
                                                    r, r, end.real, end.imag)

    def arcs(self, half_arc=False):
        """
        Return the SVG "d" path element specifiers for the two circular arcs
        about vertices A and C. If half_arc is True, the arc is at the vertex
        of a rhombus; if half_arc is False, the arc is drawn for the
        corresponding vertices of a Robinson triangle.

        """
        
        D = self.A - self.B + self.C
        arc1_d = self.get_arc_d(self.A, self.B, D, half_arc)
        arc2_d = self.get_arc_d(self.C, self.B, D, half_arc)
        return arc1_d, arc2_d

    def conjugate(self):
        """
        Return the vertices of the reflection of this triangle about the
        x-axis. Since the vertices are stored as complex numbers, we simply
        need the complex conjugate values of their values.

        """
        
        return self.__class__(self.A.conjugate(), self.B.conjugate(),
                              self.C.conjugate())

class BtileL(RobinsonTriangle):
    """
    A class representing a "B_L" Penrose tile in the P3 tiling scheme as
    a "large" Robinson triangle (sides in ratio 1:1:phi).

    """

    def inflate(self):
        """
        "Inflate" this tile, returning the three resulting Robinson triangles
        in a list.

        """

        # D and E divide sides AC and AB respectively
        D = psi2 * self.A + psi * self.C
        E = psi2 * self.A + psi * self.B
        # Take care to order the vertices here so as to get the right
        # orientation for the resulting triangles.
        return [BtileL(D, E, self.A),
                BtileS(E, D, self.B),
                BtileL(self.C, D, self.B)]

class BtileS(RobinsonTriangle):
    """
    A class representing a "B_S" Penrose tile in the P3 tiling scheme as
    a "small" Robinson triangle (sides in ratio 1:1:psi).

    """

    def inflate(self):
        """
        "Inflate" this tile, returning the two resulting Robinson triangles
        in a list.

        """
        D = psi * self.A + psi2 * self.B
        return [BtileS(D, self.C, self.A),
                BtileL(self.C, D, self.B)]

class PenroseP3:
    """ A class representing the P3 Penrose tiling. """

    def __init__(self, scale=200, ngen=4, config={}):
        """
        Initialise the PenroseP3 instance with a scale determining the size
        of the final image and the number of generations, ngen, to inflate
        the initial triangles. Further configuration is provided through the
        key, value pairs of the optional config dictionary.

        """
        
        self.scale = scale
        self.ngen = ngen

        # Default configuration
        self.config = {'width': '100%', 'height': '100%',
                       'stroke-colour': '#fff',
                       'base-stroke-width': 0.05,
                       'margin': 1.05,
                       'tile-opacity': 0.6,
                       'random-tile-colours': False,
                       'Stile-colour': '#08f',
                       'Ltile-colour': '#0035f3',
                       'Aarc-colour': '#f00',
                       'Carc-colour': '#00f',
                       'draw-tiles': True,
                       'draw-arcs': False,
                       'reflect-x': True,
                       'draw-rhombuses': True,
                       'rotate': 0,
                       'flip-y': False, 'flip-x': False,
                      }
        self.config.update(config)
        # And ensure width, height values are strings for the SVG
        self.config['width'] = str(self.config['width'])
        self.config['height'] = str(self.config['height'])

        self.elements = []

    def set_initial_tiles(self, tiles):
        self.elements = tiles

    def inflate(self):
        """ "Inflate" each triangle in the tiling ensemble."""
        new_elements = []
        for element in self.elements:
            new_elements.extend(element.inflate())
        self.elements = new_elements

    def remove_dupes(self):
        """
        Remove triangles giving rise to identical rhombuses from the
        ensemble.

        """

        # Triangles give rise to identical rhombuses if these rhombuses have
        # the same centre.
        selements = sorted(self.elements, key=lambda e: (e.centre().real,
                                                         e.centre().imag))
        self.elements = [selements[0]]
        for i, element in enumerate(selements[1:], start=1):
            if abs(element.centre() - selements[i-1].centre()) > TOL:
                self.elements.append(element)

    def add_conjugate_elements(self):
        """ Extend the tiling by reflection about the x-axis. """

        self.elements.extend([e.conjugate() for e in self.elements])

    def rotate(self, theta):
        """ Rotate the figure anti-clockwise by theta radians."""

        rot = math.cos(theta) + 1j * math.sin(theta)
        for e in self.elements:
            e.A *= rot
            e.B *= rot
            e.C *= rot

    def flip_y(self):
        """ Flip the figure about the y-axis. """

        for e in self.elements:
            e.A = complex(-e.A.real, e.A.imag)
            e.B = complex(-e.B.real, e.B.imag)
            e.C = complex(-e.C.real, e.C.imag)

    def flip_x(self):
        """ Flip the figure about the x-axis. """

        for e in self.elements:
            e.A = e.A.conjugate()
            e.B = e.B.conjugate()
            e.C = e.C.conjugate()

    def make_tiling(self):
        """ Make the Penrose tiling by inflating ngen times. """

        for gen in range(self.ngen):
            self.inflate()
        if self.config['draw-rhombuses']:
            self.remove_dupes()
        if self.config['reflect-x']:
            self.add_conjugate_elements()
            self.remove_dupes()

        # Rotate the figure anti-clockwise by theta radians.
        theta = self.config['rotate']
        if theta:
            self.rotate(theta)

        # Flip the image about the y-axis (note this occurs _after_ any
        # rotation.
        if self.config['flip-y']:
            self.flip_y()

        # Flip the image about the x-axis (note this occurs _after_ any
        # rotation and after any flip about the y-axis.
        if self.config['flip-x']:
            self.flip_x()

    def get_tile_colour(self, e):
        """ Return a HTML-style colour string for the tile. """

        if self.config['random-tile-colours']:
            # Return a random colour as '#xxx'
            return '#' + hex(random.randint(0,0xfff))[2:]

        # Return the colour string, or call the colour function as appropriate
        if isinstance(e, BtileL):
            if hasattr(self.config['Ltile-colour'], '__call__'):
                return self.config['Ltile-colour'](e)
            return self.config['Ltile-colour']

        if hasattr(self.config['Stile-colour'], '__call__'):
                return self.config['Stile-colour'](e)
        return self.config['Stile-colour']

    def filter_quadratic(self, data,condition):
        result = []
        for element in data:
            if all(condition(element,other) for other in result):
                result.append(element)
        return result

    def the_condition(self, xs,ys):
        # working with squares, 2.5e-05 is 0.005*0.005 
        return sum((x-y)*(x-y) for x,y in zip(xs,ys)) > 1e-06

    def make_target_trap(self):
        """ Make and return the data points for the tiling [(),()]. """

        draw_rhombuses = self.config['draw-rhombuses']
        penrose_points = []
        original = []
        for e in self.elements:
            if self.config['draw-tiles']:
                #original.append(e.path(rhombus=draw_rhombuses))
                # We extract the data points here.
                text = e.lufter(rhombus=draw_rhombuses)
                relative = literal_eval(re.sub("\s+", ",", text.strip()))
                #print('relative:', relative)
                absolute = []
                for i in range(len(relative)):  
                    if i == 0:
                        absolute.append(relative[i])
                    else:
                        res = tuple(map(lambda i, j: i + j, absolute[-1], relative[i]))
                        absolute.append(res)
                penrose_points.append(absolute)

        # We clean up the dulplicate points                     
        #clean_penrose_points = list(set(penrose_points[0]))
        #print('original:', original)
        #print('clean_penrose_points:', clean_penrose_points)
        # flat list is used for popout nested list
        flat_list = [item for sublist in penrose_points for item in sublist]
        clean_penrose_points = self.filter_quadratic(flat_list,self.the_condition)

        #print('clean_penrose_points:', clean_penrose_points)
        return clean_penrose_points


    def make_svg(self):
        """ Make and return the SVG for the tiling as a str. """

        xmin = ymin = -self.scale * self.config['margin']
        width =  height = 2*self.scale * self.config['margin']
        viewbox ='{} {} {} {}'.format(xmin, ymin, width, height)
        svg = ['<?xml version="1.0" encoding="utf-8"?>',
               '<svg width="{}" height="{}" viewBox="{}"'
               ' preserveAspectRatio="xMidYMid meet" version="1.1"'
               ' baseProfile="full" xmlns="http://www.w3.org/2000/svg">'
                .format(self.config['width'], self.config['height'], viewbox)]

        # The tiles' stroke widths scale with ngen
        stroke_width = str(psi**self.ngen * self.scale *
                                            self.config['base-stroke-width'])
        svg.append('<g style="stroke:{}; stroke-width: {};'
                   ' stroke-linejoin: round;">'
                .format(self.config['stroke-colour'], stroke_width))

        draw_rhombuses = self.config['draw-rhombuses']
        penrose_points = []
        original = []
        for e in self.elements:
            if self.config['draw-tiles']:
                svg.append('<path fill="{}" fill-opacity="{}" d="{}"/>'
                        .format(self.get_tile_colour(e),
                                self.config['tile-opacity'],
                                e.path(rhombus=draw_rhombuses)))

                #original.append(e.path(rhombus=draw_rhombuses))
                # We extract the data points here.
                text = e.lufter(rhombus=draw_rhombuses)
                relative = literal_eval(re.sub("\s+", ",", text.strip()))
                #print('relative:', relative)
                absolute = []
                for i in range(len(relative)):  
                    if i == 0:
                        absolute.append(relative[i])
                    else:
                        res = tuple(map(lambda i, j: i + j, absolute[-1], relative[i]))
                        absolute.append(res)
                penrose_points.append(absolute)

            if self.config['draw-arcs']:
                arc1_d, arc2_d = e.arcs(half_arc=not draw_rhombuses)
                svg.append('<path fill="none" stroke="{}" d="{}"/>'
                                .format(self.config['Aarc-colour'], arc1_d))
                svg.append('<path fill="none" stroke="{}" d="{}"/>'
                                .format(self.config['Carc-colour'], arc2_d))

        
        # We clean up the dulplicate points                     
        #clean_penrose_points = list(set(penrose_points[0]))
        #print('original:', original)
        #print('clean_penrose_points:', clean_penrose_points)
        flat_list = [item for sublist in penrose_points for item in sublist]
        clean_penrose_points = self.filter_quadratic(flat_list,self.the_condition)

        #print('clean_penrose_points:', clean_penrose_points)
        svg.append('</g>\n</svg>')
        return '\n'.join(svg), clean_penrose_points

    def write_svg(self, filename, ngen):
        """ Make and write the SVG for the tiling to filename. """
        svg, points = self.make_svg()
        with open(filename + str(ngen) + '.svg', 'w') as fo:
            fo.write(svg)

        # Write coordinates
        with open(filename + str(ngen) + '.txt', 'w') as fo:
            for element in points:
                fo.write(str(element) + "\n")

        # Make Plots
        x, y = zip(*points)
        plt.scatter(*zip(*points), c='black', s=1, marker='.')
        plt.title("Quasi-Lattice Penrose Tiling, Number of generation = {}".format(ngen))
        plt.xlim(min(x)-10, max(x)+10)
        plt.ylim(min(y)-10, max(y)+10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.savefig(filename + '.png')
        plt.show()
        
