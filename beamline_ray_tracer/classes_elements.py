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

# Matplotlib Figure parameters
FIG_HEIGHT = 8
FIG_DPI = 60


class Window:
    """
    Optical Element Class: Window
      Window Vectors:
        vectors[0] = a || length
        vectors[1] = b || width
        vectors[2] = c || normal
    :param name: str : element name
    :param position: (x,y,z) : position
    :param normal: (dx,dy,dz) : direction
    :param length: float : length perpendicular to normal along rotation axis
    :param width: float : width perpendicular to normal and rotation axis
    :param horizontal_direction: (dx,dy,dz) : in-plane direction
    """
    _debug = False

    def __init__(self, name, position, normal, length=1.0, width=1.0, horizontal_direction=None):
        self.name = name
        self.position = np.asarray(position, dtype=np.float)
        self.normal = np.asarray(normal, dtype=np.float) / np.sqrt(np.sum(np.square(normal)))
        self.length = length
        self.width = width
        self.horizontal_direction = horizontal_direction

        self.shape = ft.plane_points(position, normal, length, width, horizontal_direction)
        self.vectors = ft.plane_vectors(normal, length, width, horizontal_direction)  # [length, width, normal]
        self.size = np.sqrt(length ** 2 + width ** 2)
        self.store_interactions = []
        self.store_incident = []
        self.store_scattered = []
        self.store_normal = []  # used by curve

    def __repr__(self):
        out = 'Element(%s, position= %s, normal= %s)'
        return out % (self.name, list(self.position), list(self.normal))

    def _debug_message(self, message):
        """Display message if self._debug=True"""
        if self._debug:
            print('DBWindow: %s' % message)
    _dbm = _debug_message

    def debug(self, db=True):
        """Set debug mode"""
        self._debug = db

    def reset(self):
        """Reset beam interactions on element"""
        self.store_interactions = []
        self.store_incident = []
        self.store_scattered = []

    def gen_shape(self):
        """Generate element shape and vectors arrays"""
        self.shape = ft.plane_points(self.position, self.normal, self.length, self.width, self.horizontal_direction)
        self.vectors = ft.plane_vectors(self.normal, self.length, self.width, self.horizontal_direction)

    def move_to(self, position):
        """Change element centre to (x,y,z)"""
        self.position = np.asarray(position, dtype=np.float).reshape(3)
        self.gen_shape()

    def move_by(self, dxdydz):
        """Move element centre by (dx, dy, dz)"""
        self.position += np.asarray(dxdydz, dtype=np.float).reshape(3)
        self.gen_shape()

    def set_normal(self, normal, horizontal_direction=None):
        """Set normal direction to (dx, dy, dz)"""
        self.normal = np.asarray(normal, dtype=np.float) / np.sqrt(np.sum(np.square(normal)))
        if horizontal_direction is not None:
            self.horizontal_direction = horizontal_direction
        self.gen_shape()

    def rotate(self, angle, rotation_axis=None, rotation_centre=None):
        """
        Rotate element about an axis at a position
        :param angle: float angle in deg
        :param rotation_axis: [dx,dy,dz] direction or None to use element normal
        :param rotation_centre: [x, y, z] position or None to use element centre
        """
        if rotation_axis is None:
            rotation_axis = self.normal
        if rotation_centre is None:
            rotation_centre = self.position
        cen = np.asarray(rotation_centre, dtype=np.float)
        self.position = fg.rotate_about_axis(self.position - cen, rotation_axis, angle) + cen
        normal = fg.rotate_about_axis(self.normal, rotation_axis, angle)
        if self.horizontal_direction is not None:
            horizontal_direction = fg.rotate_about_axis(self.horizontal_direction, rotation_axis, angle)
        else:
            horizontal_direction = None
        self.set_normal(normal, horizontal_direction)

    def check_incident(self, beam_position, beam_direction):
        """Return True if beam is incident on mirror"""

        # Check beam goes anywhere near
        beam_position = np.asarray(beam_position, dtype=np.float)
        beam_direction = np.asarray(beam_direction, dtype=np.float)
        position_change = self.position - beam_position
        # max_angle = np.abs(np.tan( np.sqrt(self.length**2 + self.width**2) / fg.mag(position_change)))
        max_angle = np.pi / 2
        beam_angle = np.abs(fg.ang(position_change, beam_direction))
        # print('    angle %r: %5.2f %5.2f %s' % (self, np.rad2deg(max_angle), np.rad2deg(beam_angle), beam_angle <= max_angle))
        if beam_angle > max_angle: return False
        # dist = fg.distance2line(self.position, beam_position, beam_direction)
        # print('Line distance = %s' % dist)
        # if dist > self.size/2: return False

        # Check intersection is within points
        intersect = fg.plane_intersection(beam_position, beam_direction, self.position, self.normal)
        # print('intersect %r: %s %s' % (self, intersect, fg.isincell(intersect, self.position, self.vectors)[0]))
        if intersect is None: return False
        return fg.isincell(intersect, self.position, self.vectors)[0]

    def beam_incident(self, beam_position, beam_direction):
        """
        Find intersection of beam on element, return intersection position and new directions
        :param beam_position: [x,y,z] start point of incident beam
        :param beam_direction: [dx, dy, dz] unit vector - direction of incident beam
        :return intersect: [x,y,z] position beam intercepts the element plane
        :return direction: [ [dx,dy,dz] ] list of beam directions for outgoing beam
        """
        intersect = fg.plane_intersection(beam_position, beam_direction, self.position, self.normal)
        if not fg.isincell(intersect, self.position, self.vectors):
            return None, [None]

        self._dbm('Beam incident on window %s' % self.name)
        self._dbm('          Incident beam: %s, %s' % (beam_position, beam_direction))
        self._dbm('  Plane intersection at: %s' % intersect)

        self.store_interactions += [intersect]
        self.store_incident += [beam_direction]
        self.store_scattered += [beam_direction]
        return intersect, [beam_direction]

    def relative_position(self, position):
        """Return relative position on detector, from centre"""
        position = np.asarray(position, dtype=np.float).reshape((-1, 3))
        return fg.index_coordinates(self.position - position, self.vectors)

    def absolute_position(self, position):
        """Return position on detector in distance, from centre"""
        pos = self.relative_position(position)
        pos[:, 0] = self.length * pos[:, 0]
        pos[:, 1] = self.width * pos[:, 1]
        return pos

    def beam_width(self):
        """Return beam width in x, y"""
        if len(self.store_interactions) == 0:
            print('Beam missed %s' % self.name)
            return 0, 0
        bm = np.array(self.store_interactions)
        widx = bm[:, 0].max() - bm[:, 0].min()
        widy = bm[:, 1].max() - bm[:, 1].min()
        return widx, widy

    def beam_av_position(self):
        """Return average beam position in x, y"""
        if len(self.store_interactions) == 0:
            print('Beam missed %s' % self.name)
            return 0, 0
        bm = np.array(self.store_interactions)
        posx = np.mean(bm[:, 0])
        posy = np.mean(bm[:, 1])
        return posx, posy

    def beam_str(self):
        """Returns str giving beam interaction information"""
        if len(self.store_interactions) == 0:
            return 'Beam missed %s' % self.name
        bm = np.array(self.store_interactions)
        widx = bm[:, 0].max() - bm[:, 0].min()
        widy = bm[:, 1].max() - bm[:, 1].min()
        posx = np.mean(bm[:, 0])
        posy = np.mean(bm[:, 1])
        widx, widy = np.round(widx, 4), np.round(widy, 4)
        posx, posy = np.round(posx, 4), np.round(posy, 4)
        out = 'Beams on %s: position: (%1.5g, %1.5g), width: (%1.5g, %1.5g)'
        return out % (self.name, posx, posy, widx, widy)

    def plot_element(self):
        """Plot element in 3D"""
        fig = plt.figure(figsize=[FIG_HEIGHT, FIG_HEIGHT], dpi=FIG_DPI)
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
        # plot beams
        for n in range(len(self.store_interactions)):
            pos = [np.asarray(self.store_interactions[n]) - np.asarray(self.store_incident[n]),
                   np.asarray(self.store_interactions[n]),
                   np.asarray(self.store_interactions[n]) + np.asarray(self.store_scattered[n])]
            pos = np.array(pos)
            ax.plot(pos[:, 2], pos[:, 0], pos[:, 1], 'r-+', lw=1)

        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([-1, 1])

    def plot_projections(self):
        """Plot detector image"""
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=[FIG_HEIGHT, FIG_HEIGHT], dpi=FIG_DPI)
        plt.suptitle('Element: %s' % self.name)

        edges = self.absolute_position(self.shape)
        ax1.plot(edges[:, 0], edges[:, 2], 'k-', lw=2)
        ax2.plot(edges[:, 1], edges[:, 2], 'k-', lw=2)

        for n in range(len(self.store_interactions)):
            pos = [np.asarray(self.store_interactions[n]) - 2 * np.asarray(self.store_incident[n]),
                   np.asarray(self.store_interactions[n]),
                   np.asarray(self.store_interactions[n]) + 2 * np.asarray(self.store_scattered[n])]
            pos = self.absolute_position(pos)
            ax1.plot(pos[:, 0], pos[:, 2], 'r-+', lw=1)
            ax2.plot(pos[:, 1], pos[:, 2], 'r-+', lw=1)
            if self.store_normal:
                norm = [np.asarray(self.store_interactions[n]),
                        np.asarray(self.store_interactions[n]) - self.radius * np.asarray(self.store_normal[n])]
                norm = self.absolute_position(norm)
                ax1.plot(norm[:, 0], norm[:, 2], 'g-+', lw=1)
                ax2.plot(norm[:, 1], norm[:, 2], 'g-+', lw=1)

        ax1.axis('image')
        ax1.set_xlabel('length')
        ax1.set_ylabel('Normal')
        ax2.axis('image')
        ax2.set_xlabel('width')
        ax2.set_ylabel('Normal')

    def plot_detector_image(self, axes=None):
        """Plot detector image"""
        if axes is None:
            plt.figure(figsize=[FIG_HEIGHT, FIG_HEIGHT], dpi=FIG_DPI)
            ax = plt.subplot(111)
        else:
            ax = axes

        det = self.absolute_position(self.shape)
        pos = self.absolute_position(self.store_interactions)
        ax.plot(det[:, 0], det[:, 1], 'k-', lw=2)
        ax.plot(pos[:, 0], pos[:, 1], 'r.', ms=12)

        if axes is None:
            ax.axis('image')
            ax.set_xlabel('Detector length')
            ax.set_ylabel('Detector width')
            ax.set_title('Element: %s' % self.name)


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
            return None, [None]

        self._dbm('Beam incident on Absorber %s' % self.name)
        self._dbm('          Incident beam: %s, %s' % (beam_position, beam_direction))
        self._dbm('  Plane intersection at: %s' % intersect)

        self.store_interactions += [intersect]
        self.store_incident += [beam_direction]
        self.store_scattered += [None]
        return intersect, [None]


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
            return None, [None]

        self._dbm('Beam incident on reflector %s' % self.name)
        self._dbm('          Incident beam: %s, %s' % (beam_position, beam_direction))
        self._dbm('  Plane intersection at: %s' % intersect)

        chk = np.dot(self.normal, beam_direction)
        if abs(chk) < 0.001:
            return None, None
        elif chk > 0:
            reflected = ft.reflection(beam_direction, -self.normal)
            ang_incident = 90 + fg.ang(self.normal, beam_direction, deg=True)
            ang_reflected = -90 - fg.ang(self.normal, reflected, deg=True)
        else:
            reflected = ft.reflection(beam_direction, self.normal)
            ang_incident = 90 + fg.ang(-self.normal, beam_direction, deg=True)
            ang_reflected = -90 - fg.ang(-self.normal, reflected, deg=True)
        reflected = reflected / np.sqrt(np.sum(np.square(reflected)))
        self._dbm('         incident angle: %1.2f' % ang_incident)
        self._dbm('        reflected angle: %1.2f' % ang_reflected)
        self.store_interactions += [intersect]
        self.store_incident += [beam_direction]
        self.store_scattered += [reflected]
        return intersect, [reflected]


class CylindricalReflector(Window):
    """
    Reflector Optical element, inherits from Window
    The Reflector class reflects beams that intersect the plane.
    The scattered angle is equal to the incident angle.

    Mirror curvature is along the vertical axis (perp. to horizontal direction)
    """
    radius = 0
    origin = np.array([0, 0, 0])

    def __init__(self, name, position, normal, length=1.0, width=1.0, horizontal_direction=None, radius=1.0):
        super(CylindricalReflector, self).__init__(name, position, normal, length, width, horizontal_direction)
        self.set_radius(radius)

    def gen_shape(self):
        """Overload gen_shape to add curved lines"""
        #self.shape = ft.plane_points(self.position, self.normal, self.length, self.width, self.horizontal_direction)
        self.vectors = ft.plane_vectors(self.normal, self.length, self.width, self.horizontal_direction)
        # vectors = [vertical, horizontal, normal]

        #max_ang = np.rad2deg(np.arctan(0.5 * self.length / self.radius))
        max_ang = np.rad2deg(np.arcsin(0.5 * self.length / self.radius))
        ang = np.linspace(-max_ang, max_ang, 31)
        arc = np.array([fg.rotate_about_axis(self.radius * self.vectors[2], self.vectors[1], deg) for deg in ang])
        arc = self.origin + arc
        flat_dir = 0.5 * self.vectors[1]
        mirror_arc = np.vstack([
            arc + flat_dir,
            arc[::-1] - flat_dir,
            arc[:1] + flat_dir,
            # ft.plane_points(self.position, self.normal, self.length, self.width, self.horizontal_direction),
        ])
        self.shape = mirror_arc

    def set_normal(self, normal, horizontal_direction=None):
        """Overload set_normal to change origin"""
        self.normal = np.asarray(normal, dtype=np.float) / np.sqrt(np.sum(np.square(normal)))
        if horizontal_direction is not None:
            self.horizontal_direction = horizontal_direction
        self.set_radius()

    def set_radius(self, radius=None):
        """
        Set the curve radius
        :param radius: float: vertical radius || self.vectors[0] || length
        """
        # vertical || self.vectors[0] || length
        if radius is None:
            radius = self.radius
        else:
            self.radius = radius

        self.origin = self.position - radius * self.normal
        self.gen_shape()

    def beam_incident(self, beam_position, beam_direction):
        """return reflected beam origin and direction"""
        intersect = fg.plane_intersection(beam_position, beam_direction, self.position, self.normal)
        if not fg.isincell(intersect, self.position, self.vectors):
            return None, [None]

        chk1 = np.dot(self.normal, beam_direction)
        if abs(chk1) < 0.001:
            return None, None
        self._dbm('Beam incident on mirror %s' % self.name)
        self._dbm('          Incident beam: %s, %s' % (beam_position, beam_direction))
        self._dbm('  Plane intersection at: %s' % intersect)

        mirror_intersect, intersect_normal = ft.cylinder_intercept(beam_direction, intersect, self.origin,
                                                                   self.radius, self.vectors[2], self.vectors[0])
        chk2 = np.dot(intersect_normal, beam_direction)
        self._dbm(' Mirror intersection at: %s' % mirror_intersect)
        self._dbm('       Mirror normal at: %s' % intersect_normal)
        self._dbm('     chk1: %5.3f, chk2: %5.3f' % (chk1, chk2))

        if abs(chk2) < 0.001:
            return None, None
        elif chk2 > 0:
            reflected = ft.reflection(beam_direction, -intersect_normal)
            ang_incident = 90 + fg.ang(intersect_normal, beam_direction, deg=True)
            ang_reflected = -90 - fg.ang(intersect_normal, reflected, deg=True)
        else:
            reflected = ft.reflection(beam_direction, intersect_normal)
            ang_incident = 90 + fg.ang(-intersect_normal, beam_direction, deg=True)
            ang_reflected = -90 - fg.ang(-intersect_normal, reflected, deg=True)
        reflected = reflected / np.sqrt(np.sum(np.square(reflected)))
        self._dbm('              reflected: %s' % reflected)
        self._dbm('         incident angle: %1.2f' % ang_incident)
        self._dbm('        reflected angle: %1.2f' % ang_reflected)
        self.store_interactions += [mirror_intersect]
        self.store_incident += [beam_direction]
        self.store_scattered += [reflected]
        self.store_normal += [intersect_normal]
        return mirror_intersect, [reflected]
