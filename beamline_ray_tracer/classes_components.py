"""
beamline_ray_tracer classes
 classes of optical Components - collections of optical elements
  - Window: base class, beam passes through without alteration
  - Absorber: beam is blocked
  - Reflector: beam is reflected
"""

import numpy as np

from . import functions_general as fg
from . import functions_tracer as ft
from .classes_elements import Window, Absorber, Reflector


class Component:
    """
    Component Class
    Container of optical elements with a common relationship
    For example a curved mirror is a number of small Reflector objects
    """
    def __init__(self, name, elements, component_type=None):
        if component_type is None:
            component_type = 'Component'
        self.component_type = component_type
        self.name = name
        self.elements = elements
        self.rotation_axis = (0, 0, 1)
        self.rotation_centre = (0, 0, 0)

    def __repr__(self):
        return "%s(%s with %d elements" % (self.component_type, self.name, len(self.elements))

    def rotate(self, angle):
        for element in self.elements:
            element.rotate(angle, self.rotation_axis, self.rotation_centre)


class Detector(Component):
    """
    Detector Component
    A simple square detector using an Absorber element
    """
    def __init__(self, name, position, direction, size):
        detector = Absorber(name, position, direction, length=size, width=size)
        super(Detector, self).__init__(name, [detector], 'Detector')
        self.rotation_centre = position


class DetectorArm(Component):
    """
    DetectorArm Component
    A detector (Absorber) element rotating about a sample origin
    """
    def __init__(self, name, sample_position, delta=0, gamma=0, distance=1, length=1, width=1):
        direction = fg.you_normal_vector(0, -delta, 90-gamma)
        position = sample_position - distance * direction
        detector = Absorber(name, position, direction, length=length, width=width)
        super().__init__(name, [detector], 'Detector')
        self.rotation_centre = sample_position
        self.rotation_axis = [1, 0, 0]


class Sample(Component):
    """
    Sample Component
    A Reflector element rotating about common diffractometer angles
    """
    def __init__(self, name, position, eta=0, chi=90, phi=0, mu=0, length=1, width=1):
        normal = fg.you_normal_vector(eta, chi, mu)
        crs = fg.rotate_about_axis(np.cross(normal, [0, 0, 1]), normal, phi)
        element = Reflector(name, position, normal, length=length, width=width, horizontal_direction=crs)
        super().__init__(name, [element], 'Sample')
        self.rotation_centre = position
        self.rotation_axis = crs


class FlatMirror(Component):
    """
    Flat Mirror Component
    :param name: str : Component name
    :param position: [x,y,z] : centre position of the mirror
    :param normal: [dx,dy,dz] : nominal normal vector of the mirror
    :param pitch: float : angle in deg to rotate mirror from normal towards z
    """
    def __init__(self, name, position, normal, pitch, length=1, width=1):
        # Rotate towards z
        crs = np.cross(normal, [0, 0, 1])
        direction = fg.rotate_about_axis(normal, crs, pitch)
        element = Reflector(name, position, direction, length=length, width=width)
        super(FlatMirror, self).__init__(name, [element], 'Mirror')
        self.rotation_centre = position
        self.rotation_axis = crs


class CurvedMirrorVertical(Component):
    """
    Curved Mirror Component with curve in vertical axis
    :param name: str : Component name
    :param position: [x,y,z] : centre position of the mirror
    :param normal: [dx,dy,dz] : nominal normal vector of the mirror
    :param radius: float : radius of curvature
    :param n_elements: int : number of elements generated
    :param length: float : length of Component
    :param width: float : width of Component
    """
    def __init__(self, name, position, normal, radius, n_elements=31, length=1, width=1):
        xyz, dxdydz = ft.curved_mirror(position, normal, radius, width, n_elements, False)
        element_width = radius * 2 * np.arcsin(width / (2 * radius)) / n_elements
        elements = []
        for n in range(n_elements):
            element = Reflector(name + ': plate%d' % n, xyz[n], dxdydz[n], element_width, length, [0, 1, 0])
            elements += [element]
        super().__init__(name, elements, 'Mirror')
        self.rotation_centre = position
        self.rotation_axis = (0, 1, 0)


class CurvedMirrorHorizontal(Component):
    """
    Curved Mirror Component with curve in horizontal axis
    :param name: str : Component name
    :param position: [x,y,z] : centre position of the mirror
    :param normal: [dx,dy,dz] : nominal normal vector of the mirror
    :param radius: float : radius of curvature
    :param n_elements: int : number of elements generated
    :param length: float : length of Component
    :param width: float : width of Component
    """
    def __init__(self, name, position, normal, radius, n_elements=31, length=1, width=1):
        xyz, dxdydz = ft.curved_mirror(position, normal, radius, length, n_elements, True)
        element_length = radius * 2 * np.arcsin(length / (2 * radius)) / n_elements
        elements = []
        for n in range(n_elements):
            element = Reflector(name + ': plate%d' % n, xyz[n], dxdydz[n], element_length, width, [0, 0, 1])
            elements += [element]
        super().__init__(name, elements, 'Mirror')
        self.rotation_centre = position
        self.rotation_axis = (0, 1, 0)


class KBMirror(Component):
    """
    KB Mirrors - two curved mirrors in opposite directions
    """
    def __init__(self, name, position, beam_direction=(0, 0, 1), pitch=0, radius=1, n_elements=31, length=1, width=1):
        beam_direction = np.asarray(beam_direction, dtype=np.float)
        u1, u2, u3 = ft.plane_vectors(beam_direction)

        pos1 = np.asarray(position, dtype=np.float)
        pitch_rad = np.deg2rad(pitch)
        horiz_distance = length * np.tan(2 * pitch_rad)
        pos2 = pos1 + (length * u3) + (horiz_distance * -u2)

        dir1 = np.cos(pitch_rad) * -u2 + np.sin(pitch_rad) * -u3
        dir2 = np.cos(pitch_rad) * -u1 + np.sin(pitch_rad) * -u3

        xyz1, dxdydz1 = ft.curved_mirror(pos1, dir1, radius, length, n_elements, True)
        xyz2, dxdydz2 = ft.curved_mirror(pos2, dir2, radius, length, n_elements, True)
        element_length = radius * 2 * np.arcsin(length / (2 * radius)) / n_elements
        elements = []
        for n in range(n_elements):
            element = Reflector(name + ': vplate%d' % n, xyz1[n], dxdydz1[n], element_length, width, [0, 0, 1])
            elements += [element]
        for n in range(n_elements):
            element = Reflector(name + ': hplate%d' % n, xyz2[n], dxdydz2[n], element_length, width, [0, 0, 1])
            elements += [element]
        super().__init__(name, elements, 'KBMirror')
        self.rotation_centre = position
        self.rotation_axis = (0, 1, 0)


class ChannelCutMono(Component):
    """
    ChannelCutMono Component
    Two parallel Reflectors that rotate about a common axis
    """
    def __init__(self, name, position, bragg=0, monogap=0.1, length=1, width=1):
        direction = np.array([0, 1, 0])
        position = np.asarray(position, dtype=np.float)
        self.bragg = bragg

        plate1 = Reflector(name + ': plate1', position, direction, length=length, width=width)
        plate2 = Reflector(name + ': plate2', position+[0, monogap, length/2], direction, length=length, width=width)
        elements = [plate1, plate2]
        super(ChannelCutMono, self).__init__(name, elements, 'ChannelCutMono')
        self.rotation_centre = position
        self.rotation_axis = (1, 0, 0)
        self.rotate(-bragg)

    def rotate_bragg(self, angle_deg):
        """Set the Bragg angle"""
        angle_difference = angle_deg - self.bragg
        self.rotate(-angle_difference)
        self.bragg = angle_deg


class Slits(Component):
    """
    Slits Component
    4 Absorber elements defining a square gap. The gap size can be controlled horizontally and vertically
    """
    def __init__(self, name, position, direction, horizontal_gap=0.5, vertical_gap=0.5, plate_width=0.5):
        centre = np.asarray(position, dtype=np.float)
        direction = np.asarray(direction, dtype=np.float)
        self.horizontal_gap = horizontal_gap
        self.vertical_gap = vertical_gap
        self.plate_width = plate_width

        window = Window(name + ': gap', centre, direction, length=horizontal_gap, width=vertical_gap)
        plate1_pos = centre + (plate_width / 2) * fg.norm(window.vectors[0]) + window.vectors[0] / 2
        plate2_pos = centre - (plate_width / 2) * fg.norm(window.vectors[0]) - window.vectors[0] / 2
        plate3_pos = centre + (plate_width / 2) * fg.norm(window.vectors[1]) + window.vectors[1] / 2
        plate4_pos = centre - (plate_width / 2) * fg.norm(window.vectors[1]) - window.vectors[1] / 2

        plate1 = Absorber(name + ': hplate1', plate1_pos, direction, length=plate_width, width=plate_width)
        plate2 = Absorber(name + ': hplate2', plate2_pos, direction, length=plate_width, width=plate_width)
        plate3 = Absorber(name + ': vplate1', plate3_pos, direction, length=plate_width, width=plate_width)
        plate4 = Absorber(name + ': vplate2', plate4_pos, direction, length=plate_width, width=plate_width)
        elements = [window, plate1, plate2, plate3, plate4]

        super().__init__(name, elements, 'Slits')
        self.rotation_centre = position
        self.rotation_axis = (0, 1, 0)

    def set_hgap(self, horizontal_gap):
        """Set horizontal gap"""
        current_hgap = self.horizontal_gap
        gap_difference = horizontal_gap - current_hgap

        plate1 = self.elements[1]
        plate2 = self.elements[2]

        newpos1 = plate1.position + (gap_difference / 2) * plate1.vectors[0]
        newpos2 = plate2.position + (gap_difference / 2) * plate2.vectors[0]
        plate1.position = newpos1
        plate2.position = newpos2
        self.horizontal_gap = horizontal_gap

    def set_vgap(self, vertical_gap):
        """Set vertical gap"""
        current_vgap = self.horizontal_gap
        gap_difference = vertical_gap - current_vgap

        plate1 = self.elements[3]
        plate2 = self.elements[4]

        newpos1 = plate1.position + (gap_difference / 2) * plate1.vectors[1]
        newpos2 = plate2.position + (gap_difference / 2) * plate2.vectors[1]
        plate1.position = newpos1
        plate2.position = newpos2
        self.vertical_gap = vertical_gap
