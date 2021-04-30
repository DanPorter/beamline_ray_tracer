"""
Class Beam
"""

import numpy as np
import matplotlib.pyplot as plt

from .classes_elements import FIG_HEIGHT, FIG_DPI
from . import functions_general as fg


class Beam:
    """
    Beam Object
    Defines a light beam that interacts with optical elements
    Contains a list of positions and a current direction

    beam = Beam(x,y,z, dx,dy,dz)
     x,y,z = initial position
     dx,dy,dz = initial direction

    beam.new( (x,y,z), (dx,dy,dz) ) - adds new position and changes direction

    """

    def __init__(self, x=0, y=0, z=0, dx=0, dy=0, dz=1, beam_id=0):
        self.x, self.y, self.z = x, y, z
        self.dx, self.dy, self.dz = dx, dy, dz
        self.positions = [np.array([x, y, z], dtype=np.float)]
        direction = np.asarray([dx, dy, dz], dtype=np.float)
        self.direction = direction/np.sqrt(np.sum(np.square(direction)))
        self.beam_id = beam_id
        # Future parameters:
        self.energy_kev = 8.0
        self.intensity = 100.0
        self.polarisation = (1, 0, 0)  # P1, P2, P3 Poincare-Stokes parameters
        self.phase = 0.0  # 0-2pi

    def __repr__(self):
        current_pos, current_dir = self.current()
        return "Beam(%2d, pos: %s, dir: %s)" % (self.beam_id, current_pos, current_dir)

    def reset(self):
        """Reset Beam to original position & direction"""
        self.positions = [np.array([self.x, self.y, self.z], dtype=np.float)]
        direction = np.asarray([self.dx, self.dy, self.dz], dtype=np.float)
        self.direction = direction / np.sqrt(np.sum(np.square(direction)))

    def xyz(self):
        """Return array of positions"""
        return np.array(self.positions)

    def add_position(self, x, y, z):
        """Add a new position"""
        self.positions += [np.array([x, y, z], dtype=np.float)]

    def new_direction(self, dx, dy, dz):
        """Define the direction after the last point"""
        direction = np.asarray([dx, dy, dz], dtype=np.float)
        self.direction = direction/np.sqrt(np.sum(np.square(direction)))

    def current(self):
        """Return current position and direction"""
        return self.positions[-1], self.direction

    def new(self, position, direction):
        """Add a new beam path"""
        self.add_position(*position)
        if direction is None:
            self.direction = None
        else:
            self.new_direction(*direction)

    def extrude(self, distance=1.0):
        """Add a new beam position along current direction"""
        pos, direction = self.current()
        if direction is None:
            return
        new = pos + distance*direction
        self.add_position(*new)

    def total_distance(self):
        """Calculate total distance travelled by beam"""
        distance = 0
        for n in range(1, len(self.positions)):
            distance += fg.mag(self.positions[n] - self.positions[n-1])
        return distance

    def intersect_xy(self, beam):
        """Returns the intersection position with another beam in the xy plane"""
        pos1 = self.positions[-1][[0, 1]]
        dir1 = self.direction[[0, 1]]
        pos2 = beam.positions[-1][[0, 1]]
        dir2 = beam.direction[[0, 1]]
        return fg.vector_intersection(pos1, dir1, pos2, dir2)

    def intersect_xz(self, beam):
        """Returns the intersection position with another beam in the xz plane"""
        pos1 = self.positions[-1][[0, 2]]
        dir1 = self.direction[[0, 2]]
        pos2 = beam.positions[-1][[0, 2]]
        dir2 = beam.direction[[0, 2]]
        return fg.vector_intersection(pos1, dir1, pos2, dir2)

    def intersect_yz(self, beam):
        """Returns the intersection position with another beam in the yz plane"""
        pos1 = self.positions[-1][[1, 2]]
        dir1 = self.direction[[1, 2]]
        pos2 = beam.positions[-1][[1, 2]]
        dir2 = beam.direction[[1, 2]]
        return fg.vector_intersection(pos1, dir1, pos2, dir2)

    def intersect(self, beam):
        """Returns the point of intersectin with another beam in 3D"""
        pos1 = self.positions[-1]
        dir1 = self.direction
        pos2 = beam.positions[-1]
        dir2 = beam.direction
        intersect = fg.vector_intersection3d(pos1, dir1, pos2, dir2)
        distance = fg.distance2line(pos1, pos1 + dir1, intersect)
        if distance > 1e-6:
            print('Vectors dont intersect')
        return intersect, distance

    def angle(self, beam, deg=True):
        """Returns the angle in degrees between who beams"""
        return fg.ang(self.direction, beam.direction, deg)


class Interaction:
    """
    Beam interaction on optical element
    :param position: (x, y, z)
    :param incident: (dx, dy, dz)
    :param scattered: (dx, dy, dz)
    """

    def __init__(self, position, incident, scattered):
        self.position = np.asarray(position, dtype=np.float).reshape(3)
        self.incident = np.asarray(incident, dtype=np.float) / np.sqrt(np.sum(np.square(incident)))
        self.scattered = np.asarray(scattered, dtype=np.float) / np.sqrt(np.sum(np.square(scattered)))

    def plot(self, axes=None, length=1):
        if axes is None:
            fig = plt.figure(figsize=[FIG_HEIGHT, FIG_HEIGHT], dpi=FIG_DPI)
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axes

        pos = [self.position - length * self.incident,
               self.position,
               self.position + length * self.scattered]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'r-+', lw=1)

        if axes is None:
            ax.set_xlabel('z')
            ax.set_ylabel('x')
            ax.set_zlabel('y')
            # ax.set_xlim([-1, 1])
            # ax.set_ylim([-1, 1])
            # ax.set_zlim([-1, 1])