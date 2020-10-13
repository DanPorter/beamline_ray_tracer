"""
Class Beam
"""

import numpy as np

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
        self.positions = [np.array([x, y, z], dtype=np.float)]
        direction = np.asarray([dx, dy, dz], dtype=np.float)
        self.direction = direction/np.sqrt(np.sum(np.square(direction)))
        self.beam_id = beam_id

    def __repr__(self):
        current_pos, current_dir = self.current()
        return "Beam(%2d, pos: %s, dir: %s)" % (self.beam_id, current_pos, current_dir)

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
