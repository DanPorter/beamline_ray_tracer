"""
beamline_ray_tracer classes
 classes of optical elements - base interaction elements
  - Window: base class, beam passes through without alteration
  - Absorber: beam is blocked
  - Reflector: beam is reflected
"""

import numpy as np
import matplotlib.pyplot as plt

from . import functions_general as fg
from . import functions_tracer as ft


class Window:
    """
    Optical Element Class: Window
    :param name: str : element name
    :param position: (x,y,z) : position
    :param normal: (dx,dy,dz) : direction
    :param length: float : length perpendicular to normal along rotation axis
    :param width: float : width perpendicular to normal and rotation axis
    :param horizontal_direction:
    """
    _debug = True

    def __init__(self, name, position, normal, length=1.0, width=1.0, horizontal_direction=None):
        self.name = name
        self.position = np.asarray(position, dtype=np.float)
        self.normal = np.asarray(normal, dtype=np.float)/np.sqrt(np.sum(np.square(normal)))
        self.length = length
        self.width = width
        self.horizontal_direction = horizontal_direction

        self.shape = ft.plane_points(position, normal, length, width, horizontal_direction)
        self.vectors = ft.plane_vectors(normal, length, width, horizontal_direction)
        self.size = np.sqrt(length**2 + width**2)
        self.store_interactions = []

    def __repr__(self):
        out = 'Element(%s, position= %s, normal= %s)'
        return out % (self.name, list(self.position), list(self.normal))

    def gen_shape(self):
        self.shape = ft.plane_points(self.position, self.normal, self.length, self.width, self.horizontal_direction)
        self.vectors = ft.plane_vectors(self.normal, self.length, self.width, self.horizontal_direction)

    def rotate(self, angle, rotation_axis, rotation_centre):
        cen = np.asarray(rotation_centre, dtype=np.float)
        self.position = fg.rotate_about_axis(self.position-cen, rotation_axis, angle) + cen
        self.normal = fg.rotate_about_axis(self.normal, rotation_axis, angle)
        self.gen_shape()

    def check_incident(self, beam_position, beam_direction):
        """Return True if beam is incident on mirror"""

        # Check beam goes anywhere near
        beam_position = np.asarray(beam_position, dtype=np.float)
        beam_direction = np.asarray(beam_direction, dtype=np.float)
        position_change = self.position - beam_position
        #max_angle = np.abs(np.tan( np.sqrt(self.length**2 + self.width**2) / fg.mag(position_change)))
        max_angle = np.pi/2
        beam_angle = np.abs(fg.ang(position_change, beam_direction))
        #print('    angle %r: %5.2f %5.2f %s' % (self, np.rad2deg(max_angle), np.rad2deg(beam_angle), beam_angle <= max_angle))
        if beam_angle > max_angle: return False
        #dist = fg.distance2line(self.position, beam_position, beam_direction)
        #print('Line distance = %s' % dist)
        #if dist > self.size/2: return False

        # Check intersection is within points
        intersect = fg.plane_intersection(beam_position, beam_direction, self.position, self.normal)
        #print('intersect %r: %s %s' % (self, intersect, fg.isincell(intersect, self.position, self.vectors)[0]))
        if intersect is None: return False
        return fg.isincell(intersect, self.position, self.vectors)[0]

    def beam_incident(self, beam_position, beam_direction):
        """return transmitted beam"""
        intersect = fg.plane_intersection(beam_position, beam_direction, self.position, self.normal)
        if not fg.isincell(intersect, self.position, self.vectors):
            return None, None
        self.store_interactions += [intersect]
        return intersect, beam_direction

    def relative_position(self, position):
        """Return relative position on detector, from centre"""
        position = np.asarray(position, dtype=np.float).reshape((-1, 3))
        return fg.index_coordinates(position - self.position, self.vectors)

    def plot_element(self):
        """Plot element in 3D"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.shape[:, 2], self.shape[:, 0], self.shape[:, 1], 'k-')
        label = ['length', 'width', 'normal']
        for n in range(3):
            ax.plot([self.position[2], self.position[2] + self.vectors[n][2]],
                    [self.position[0], self.position[0] + self.vectors[n][0]],
                    [self.position[1], self.position[1] + self.vectors[n][1]],
                    'k-', lw=0.5)
            ax.text(self.position[2] + self.vectors[n][2],
                    self.position[0] + self.vectors[n][0],
                    self.position[1] + self.vectors[n][1],
                    label[n])

        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        #ax.set_xlim([-1, 1])
        #ax.set_ylim([-1, 1])
        #ax.set_zlim([-1, 1])

    def plot_detector_image(self):
        """Plot detector image"""
        plt.figure()
        det = self.relative_position(self.shape)
        pos = self.relative_position(np.array(self.store_interactions))
        plt.plot(self.length * det[:, 0], self.width * det[:, 1], 'k-', lw=2)
        plt.plot(self.length * pos[:, 0], self.width * pos[:, 1], 'rx', ms=12)
        plt.axis('image')
        plt.xlabel('Detector length')
        plt.ylabel('Detector width')
        plt.title('Absorber: %s' % self.name)


class Absorber(Window):
    """
    Absorber Optical element, inherits from Window
    The Absorber class stops beams that intersect the plane.
    An beam_indident call will return the intersecting position and None for direction
    """
    def beam_incident(self, beam_position, beam_direction):
        """return transmitted beam"""
        intersect = fg.plane_intersection(beam_position, beam_direction, self.position, self.normal)
        if not fg.isincell(intersect, self.position, self.vectors):
            return None, None
        self.store_interactions += [intersect]
        return intersect, None


class Reflector(Window):
    """
    Reflector Optical element, inherits from Window
    The Reflector class reflects beams that intersect the plane.
    The scattered angle is equal to the incident angle.
    """
    def beam_incident(self, beam_position, beam_direction):
        """return reflected beam origin and direction"""
        intersect = fg.plane_intersection(beam_position, beam_direction, self.position, self.normal)
        if not fg.isincell(intersect, self.position, self.vectors):
            return None, None

        chk = np.dot(self.normal, beam_direction)
        if abs(chk) < 0.001:
            return None, None
        elif chk > 0:
            reflected = ft.reflection(beam_direction, -self.normal)
            ang_incident = 90+fg.ang(self.normal, beam_direction, deg=True)
            ang_reflected = -90-fg.ang(self.normal, reflected, deg=True)
        else:
            reflected = ft.reflection(beam_direction, self.normal)
            ang_incident = 90+fg.ang(-self.normal, beam_direction, deg=True)
            ang_reflected = -90-fg.ang(-self.normal, reflected, deg=True)
        reflected = reflected/np.sqrt(np.sum(np.square(reflected)))
        # print('Reflect on: %s' % self.name)
        # print('  incident angle: %1.2f' % ang_incident)
        # print(' reflected angle: %1.2f' % ang_reflected)
        self.store_interactions += [intersect]
        return intersect, reflected


