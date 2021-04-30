"""
beamline_ray_tracer classes
 classes of optical Components - collections of optical elements
  - Window: base class, beam passes through without alteration
  - Absorber: beam is blocked
  - Reflector: beam is reflected
"""

import numpy as np
import matplotlib.pyplot as plt

from . import functions_general as fg
from . import functions_tracer as ft
from .classes_elements import Window, Absorber, Reflector, CylindricalReflector, FIG_HEIGHT, FIG_DPI


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
        return "%s(%s with %d elements)" % (self.component_type, self.name, len(self.elements))

    def __str__(self):
        out = '%s\n' % self.__repr__()
        out += '\n'.join('  %s' % el.__repr__() for el in self.elements)
        return out

    def __getitem__(self, item):
        return self.elements[item]

    def debug(self, db=True):
        """Set debug mode on all elements"""
        for el in self.elements:
            el.debug(db)

    def move(self, dxdydz):
        """Move all element centres by (dx, dy, dz)"""
        for element in self.elements:
            element.move_by(dxdydz)

    def rotate(self, angle):
        for element in self.elements:
            element.rotate(angle, self.rotation_axis, self.rotation_centre)

    def plot(self, axes=None):
        """ Plot Beam path and Optical Elements in 3d """
        if axes is None:
            fig = plt.figure(figsize=[FIG_HEIGHT, FIG_HEIGHT], dpi=FIG_DPI)
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axes

        for element in self.elements:
            ax.plot(element.shape[:, 2], element.shape[:, 0], element.shape[:, 1], 'k-')
            #ax.plot([element.position[2], element.position[2]+element.normal[2]],
            #        [element.position[0], element.position[0] + element.normal[0]],
            #        [element.position[1], element.position[1] + element.normal[1]],
            #        'k-', lw=0.5)

            for n in range(len(element.store_interactions)):
                pos = [np.asarray(element.store_interactions[n]) - np.asarray(element.store_incident[n]),
                       np.asarray(element.store_interactions[n]),
                       np.asarray(element.store_interactions[n]) + np.asarray(element.store_scattered[n])]
                pos = np.array(pos)
                ax.plot(pos[:, 2], pos[:, 0], pos[:, 1], 'r-+', lw=1)

        if axes is None:
            ax.set_xlabel('z')
            ax.set_ylabel('x')
            ax.set_zlabel('y')
            ax.set_title(self.name)
            #ax.set_xlim([-1, 1])
            #ax.set_ylim([-1, 1])
            #ax.set_zlim([-1, 1])
            plt.show()

    def plot_projections(self, image=False):
        """ Plot Beam path and Optical Elements as 2d projections """
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=[2 * FIG_HEIGHT, FIG_HEIGHT], dpi=FIG_DPI)
        plt.suptitle('%r' % self)
        #  x vs z
        for element in self.elements:
            ax1.plot(element.shape[:, 0], element.shape[:, 2], 'k-')
            ax2.plot(element.shape[:, 1], element.shape[:, 2], 'k-')
            ax3.plot(element.shape[:, 0], element.shape[:, 1], 'k-')

            for n in range(len(element.store_interactions)):
                pos = np.array([
                    np.asarray(element.store_interactions[n]) - 2 * np.asarray(element.store_incident[n]),
                    np.asarray(element.store_interactions[n]),
                    np.asarray(element.store_interactions[n]) + 2 * np.asarray(element.store_scattered[n])
                ])
                ax1.plot(pos[:, 0], pos[:, 2], 'r-+', lw=1)
                ax2.plot(pos[:, 1], pos[:, 2], 'r-+', lw=1)
                ax3.plot(pos[:, 0], pos[:, 1], 'r-+', lw=1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('z')
        ax2.set_xlabel('y')
        ax2.set_ylabel('z')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        if image:
            ax1.axis('image')
            ax2.axis('image')
            ax3.axis('image')


class Detector(Component):
    """
    Detector Component
    A simple square detector using an Absorber element
    """
    def __init__(self, name, position, direction, size):
        detector = Absorber(name, position, direction, length=size, width=size)
        super(Detector, self).__init__(name, [detector], 'Detector')
        self.rotation_centre = position


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
        crs = np.cross(normal, [0, 0, 1])  # pitch will rotate about the [1,0,0] axis
        if fg.mag(crs) < 0.1:
            crs = np.cross(normal, [1, 0, 0]) # pitch will rotate about the [0,0,1] axis
        direction = fg.rotate_about_axis(normal, crs, pitch)
        element = Reflector(name, position, direction, length=length, width=width)
        super().__init__(name, [element], 'Mirror')
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
    :param length: float : length of Component (along beam direction)
    :param width: float : width of Component (perpendicular to beam direction and normal)
    """
    def __init__(self, name, position, normal, radius, n_elements=31, length=1, width=1):
        self.position = np.asarray(position, dtype=np.float).reshape(3)
        self.normal = np.asarray(normal, dtype=np.float).reshape(3)
        self.radius = radius
        self.curve_size = width
        xyz, dxdydz = ft.curved_mirror(position, normal, radius, width, n_elements, False)
        element_width = radius * 2 * np.arcsin(width / (2 * radius)) / (n_elements-1)
        elements = []
        for n in range(n_elements):
            #element = Reflector(name + ': plate%d' % n, xyz[n], dxdydz[n], element_width, length, [0, 1, 0])
            element = CylindricalReflector(name + ': plate%d' % n, xyz[n], -dxdydz[n], element_width, length,
                                           horizontal_direction=[0, 1, 0], radius=radius)

            elements += [element]
        super().__init__(name, elements, 'Mirror')
        self.rotation_centre = position
        self.rotation_axis = (0, 1, 0)

    def move(self, dxdydz):
        self.position += np.asarray(dxdydz, dtype=np.float).reshape(3)
        super().move(dxdydz)

    def rotate(self, angle):
        self.normal = fg.rotate_about_axis(self.normal, self.rotation_axis, angle)
        super().rotate(angle)

    def element_size(self, radius=None):
        """Returns the width of an individual mirror element along the axis of curvature"""
        if radius is None:
            radius = self.radius
        n_elements = len(self.elements)
        return radius * 2 * np.arcsin(self.curve_size / (2 * radius)) / (n_elements - 1)

    def max_angle_error(self):
        """Returns the maximum angular deviation given the number of elements used"""
        return np.rad2deg(np.arctan(self.element_size()/(2*self.radius)))

    def max_distance_error(self):
        """Returns the maximum distance deviation given the number of elements used"""
        return np.sqrt((self.element_size()/2)**2 + self.radius**2) - self.radius

    def change_radius(self, radius):
        """Move + rotate plates to change mirror radius"""
        n_elements = len(self.elements)
        xyz, dxdydz = ft.curved_mirror(self.position, self.normal, radius, self.curve_size, n_elements, False)
        element_width = radius * 2 * np.arcsin(self.curve_size / (2 * radius)) / (n_elements - 1)
        for n, element in enumerate(self.elements):
            element.width = element_width
            element.move_to(xyz[n, :])
            element.set_normal(dxdydz[n, :])


class CurvedMirrorHorizontal(Component):
    """
    Curved Mirror Component with curve in horizontal axis
    :param name: str : Component name
    :param position: [x,y,z] : centre position of the mirror
    :param normal: [dx,dy,dz] : nominal normal vector of the mirror
    :param radius: float : radius of curvature (curve in the plane of beam direction and normal)
    :param n_elements: int : number of elements generated
    :param length: float : length of Component (along beam direction)
    :param width: float : width of Component (perpendicular to beam direction and normal)
    """
    def __init__(self, name, position, normal, radius, n_elements=31, length=1, width=1):
        self.position = np.asarray(position, dtype=np.float).reshape(3)
        self.normal = np.asarray(normal, dtype=np.float).reshape(3)
        self.radius = radius
        self.curve_size = length
        xyz, dxdydz = ft.curved_mirror(position, normal, radius, length, n_elements, True)
        self.focal_position = xyz[0] + radius*dxdydz[0]
        element_length = radius * 2 * np.arcsin(length / (2 * radius)) / (n_elements-1)
        elements = []
        for n in range(n_elements):
            #element = Reflector(name + ': plate%d' % n, xyz[n], dxdydz[n], element_length, width, [0, 0, 1])
            element = CylindricalReflector(name + ': plate%d' % n, xyz[n], -dxdydz[n], element_length, width, [1, 0, 0],
                                           radius=radius)
            elements += [element]
        super().__init__(name, elements, 'Mirror')
        self.rotation_centre = position
        self.rotation_axis = (0, 1, 0)

    def move(self, dxdydz):
        self.position += np.asarray(dxdydz, dtype=np.float).reshape(3)
        super().move(dxdydz)

    def rotate(self, angle):
        self.normal = fg.rotate_about_axis(self.normal, self.rotation_axis, angle)
        super().rotate(angle)

    def element_size(self, radius=None):
        """Returns the width of an individual mirror element along the axis of curvature"""
        if radius is None:
            radius = self.radius
        n_elements = len(self.elements)
        return radius * 2 * np.arcsin(self.curve_size / (2 * radius)) / (n_elements - 1)

    def max_angle_error(self):
        """Returns the maximum angular deviation given the number of elements used"""
        return np.rad2deg(np.arctan(self.element_size() / (2 * self.radius)))

    def max_distance_error(self):
        """Returns the maximum distance deviation given the number of elements used"""
        return np.sqrt((self.element_size() / 2) ** 2 + self.radius ** 2) - self.radius

    def change_radius(self, radius):
        """Move + rotate plates to change mirror radius"""
        n_elements = len(self.elements)
        xyz, dxdydz = ft.curved_mirror(self.position, self.normal, radius, self.curve_size, n_elements, True)
        element_length = radius * 2 * np.arcsin(self.curve_size / (2 * radius)) / (n_elements-1)
        for n, element in enumerate(self.elements):
            element.width = element_length
            element.move_to(xyz[n, :])
            element.set_normal(dxdydz[n, :])


class MirrorSystem(Component):
    """
    Double Mirror system
    Fixed focus vertical cyclindrical primary mirror followed by variable radius horizontally horizontal mirror (bender)
    Vertical focus is controlled by the matched mirror pitch, horizontal focus is controlled by the radius of the
    bender.
    """
    def __init__(self, name, position, pitch=0, m1m2distance=1, m1_radius=0.01, m2_radius=1000,
                 n_elements=31, length=1, width=1):
        self.pitch = pitch
        self.m1m2distance = m1m2distance  # Mirror separation parallel to beam
        self.m1_position = np.asarray(position, dtype=np.float).reshape(3)
        self.m1_radius = m1_radius  # sagittal (vertical)
        self.m2_radius = m2_radius  # meridonal (horizontal)

        normal = np.asarray((1, 0, 0), dtype=np.float)
        parallel = np.asarray((0, 0, 1), dtype=np.float)
        self.normal = normal
        self.parallel = parallel
        separation = m1m2distance * np.tan(2 * np.deg2rad(pitch))  # mirror separation perpendicular to beam
        self.m2_position = self.m1_position + m1m2distance * parallel + separation * normal

        self.m1 = CurvedMirrorVertical('Cyclindrical', position, normal, radius=m1_radius, n_elements=n_elements,
                                       length=length, width=width)
        self.m2 = CurvedMirrorHorizontal('Bender', self.m2_position, -normal, radius=m2_radius, n_elements=n_elements,
                                         length=length, width=width)
        self.m1.rotate(pitch)
        self.m2.rotate(pitch)
        super().__init__(name, self.m1.elements + self.m2.elements, 'Mirror')
        self.rotation_centre = position
        self.rotation_axis = (0, 1, 0)

    def focal_lengths(self):
        """Calculate focal distances from m1 and m2, distance returned from m1 position"""
        vertical_focus = ft.sagittal_focal_length(self.pitch, self.m1_radius)
        horizontal_focus = ft.meridonal_focal_length(self.pitch, self.m2_radius)
        # add distance between mirrors
        horizontal_focus += self.m1m2distance
        # Technically I should also subtract a little from vertical
        return vertical_focus, horizontal_focus

    def change_pitch(self, pitch):
        """Alter the pitch of both mirrors"""
        self.m1.rotate(pitch - self.pitch)
        self.m2.rotate(pitch - self.pitch)
        separation = self.m1m2distance * np.tan(2 * np.deg2rad(pitch))  # mirror separation perpendicular to beam
        old_position = self.m2_position
        new_position = self.m1_position + self.m1m2distance * self.parallel + separation * self.normal
        difference = new_position - old_position
        self.m2.move(difference)
        self.pitch = pitch
        self.m2_position = new_position

    def change_bender(self, radius):
        """Change m2 horizontal radius"""
        self.m2.change_radius(radius)


class KBMirror(Component):
    """
    KB Mirrors - two curved mirrors in opposite directions
    :param name: str : Component name
    :param position: [x,y,z] : centre position of the mirror
    :param beam_direction: [dx,dy,dz] : direction of incident beam
    :param pitch: float : mirror pitch angle for focussing
    :param radius: float : mirror radius
    :param n_elements: int : number of mirror elements.
    :param length: float : length of Component
    :param width: float : width of Component
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
        element_length = radius * 2 * np.arcsin(length / (2 * radius)) / (n_elements-1)
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
    :param name: str : Component name
    :param position: [x,y,z] : centre position of the mirror
    :param bragg: float : pitch angle defining scattering (Bragg) angle
    :param monogap: float : distance between plates
    :param d_space: float : diffraction d_space for calcualtion of Bragg angle (default Si (111))
    :param length: float : length of Component
    :param width: float : width of Component
    """
    def __init__(self, name, position, bragg=0, monogap=0.1, d_space=3.1356, length=1, width=1):
        direction = np.array([0, 1, 0])
        position = np.asarray(position, dtype=np.float)
        self.bragg = bragg
        self.d_space = d_space  # 5.431020511 / np.sqrt(3)  # Silicon (111)

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

    def rotate_energy(self, energy_kev):
        """Set the Bragg angle for a given energy in keV, assuming Si(111) reflection"""
        # n*lambda = 2d sin theta
        bragg = np.arcsin(6.19922 / (energy_kev * self.d_space))
        bragg_deg = np.rad2deg(bragg)
        self.rotate_bragg(bragg_deg)

    def rotate_wavelength(self, wl_angstrom):
        """Set the Bragg angle for a given wavelength in angstrom, assuming Si(111) reflection"""
        # n*lambda = 2d sin theta
        bragg = np.arcsin(wl_angstrom / (2 * self.d_space))
        bragg_deg = np.rad2deg(bragg)
        self.rotate_bragg(bragg_deg)


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
