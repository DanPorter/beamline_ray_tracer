"""
beamline_ray_tracer classes
Room Class
 - defines the packaging of optical elements and beams, contains the main program loop
"""

import numpy as np
import matplotlib.pyplot as plt

from .classes_elements import Reflector, Absorber, FIG_HEIGHT, FIG_DPI
from .classes_beam import Beam
from .functions_tracer import plane_vectors


class Room:
    """
    Room containing optical elements - container for Beam and Element objects
      room = Room('Optical Bench')
      room.generate_beams()
      room.add_reflector('mirror', (0,0,1), (0,1,-1))
      room.add_absorber('detector', (0,1,1), (0,-1,0))
      room.run()
      room.plot()
    """
    _MAX_ITER = 100

    def __init__(self, name):
        self.name = name
        self.beams = []
        self.elements = []

    def __repr__(self):
        return "Room(%s with %d beams and %d elements)" % (self.name, len(self.beams), len(self.elements))

    def __str__(self):
        out = self.__repr__()
        out += '\nBeams:\n'
        out += '\n'.join('%r' % beam for beam in self.beams)
        out += '\nElements:\n'
        out += '\n'.join('%r' % el for el in self.elements)
        return out

    def add_beam(self, position=(0, 0, 0), direction=(0, 0, 1)):
        """Add beam at given position in given direction"""
        new_id = len(self.beams) + 1
        x, y, z = position
        dx, dy, dz = direction
        self.beams += [Beam(x, y, z, dx, dy, dz, new_id)]

    def generate_beams(self, position=(0, 0, 0), direction=(0, 0, 1),
                       horizontal_width=0.5, vertical_width=0.5, n_beams=5):
        """
        generate diamond shape of beams with different widths
        Will generate n_beams * n_beams / 2 beam objects
        :param position: tuple : (x, y, z) : centre position
        :param direction: tuple : (x, y, z) : beam direction
        :param horizontal_width: float : max width of beams normal to direction
        :param vertical_width: flaot : max width of beams normal to direction
        :param n_beams: int : number of beams wide (same number high)
        :return: None
        """
        position = np.asarray(position, dtype=np.float)
        a, b, c = plane_vectors(direction, horizontal_width, vertical_width)
        for v in np.arange(n_beams) - n_beams//2:
            vrange = n_beams - 2*np.abs(v)
            for h in np.arange(vrange) - vrange//2:
                pos = position + 0.5 * h * a / (n_beams//2) + 0.5 * v * b / (n_beams//2)
                self.add_beam(pos, direction)

    def add_element_list(self, element_list):
        """Add optical element to room list"""
        self.elements += element_list

    def add_reflector(self, name, position, normal, length=1.0, width=1.0):
        """Add Reflector optical element to room list"""
        self.elements += [Reflector(name, position, normal, length, width)]

    def add_absorber(self, name, position, normal, length=1.0, width=1.0):
        """Add Absorber optical element to room list"""
        self.elements += [Absorber(name, position, normal, length, width)]

    def add_component(self, component):
        """Add optical elements from a component to the room list"""
        self.elements += component.elements

    def ordered_elements(self, position, current_element=None):
        """Return list of elements ordered by distance to position=(x,y,z)"""
        position = np.asarray(position, dtype=np.float)

        elements = [el for el in self.elements if el is not current_element]
        distances = np.array([np.sqrt(np.sum(np.square(position - el.position))) for el in elements])
        sort_index = np.argsort(distances)
        return [elements[n] for n in sort_index]

    def next_element(self, beam, current_element=None):
        """Determine the next element to scatter from"""
        beam_pos, beam_dir = beam.current()
        if beam_dir is None: return None
        sort_elements = self.ordered_elements(beam_pos, current_element)
        for el in sort_elements:
            if el.check_incident(beam_pos, beam_dir):
                return el
        return None

    def reset(self):
        """Reset beams"""
        for beam in self.beams:
            beam.reset()
        for element in self.elements:
            element.reset()

    def run(self, debug=False):
        """Run the Ray tracing program"""

        if len(self.beams) == 0:
            print("Add Beams first")
            return
        if len(self.elements) == 0:
            print("Add some optical elements")
            return

        if debug:
            print('\nRunning Ray Tracing for %s' % self.__repr__())

        self.reset()

        for beam in self.beams:
            if debug:
                print('\n---Beam %d---' % beam.beam_id)
            element = self.next_element(beam)
            iter_number = 0
            while element is not None and iter_number < self._MAX_ITER:
                iter_number += 1
                current_pos, current_dir = beam.current()
                if current_dir is None: break

                new_pos, new_dir = element.beam_incident(current_pos, current_dir)
                if new_pos is None: break
                if debug:
                    print('%r' % element)
                    print('   current beam: %r' % beam)
                beam.new(new_pos, new_dir)
                if debug:
                    print(' reflected beam:%r\n' % beam)
                element = self.next_element(beam, element)
                if debug:
                    print('Next element: %r\n' % element)
            # If no element, continue beam for a while
            if beam.direction is not None:
                beam.extrude(1.0)

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

        for beam in self.beams:
            pos = beam.xyz()
            ax.plot(pos[:, 2], pos[:, 0], pos[:, 1], 'b-o', lw=2, ms=4)

        if axes is None:
            ax.set_xlabel('z')
            ax.set_ylabel('x')
            ax.set_zlabel('y')
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            plt.show()

    def plot_projections(self):
        """ Plot Beam path and Optical Elements as 2d projections """
        fig = plt.figure(figsize=[2 * FIG_HEIGHT, FIG_HEIGHT], dpi=FIG_DPI)

        #  x vs z
        ax = fig.add_subplot(131)
        for element in self.elements:
            ax.plot(element.shape[:, 0], element.shape[:, 2], 'k-')
        for beam in self.beams:
            pos = beam.xyz()
            ax.plot(pos[:, 0], pos[:, 2], 'b-o', lw=2, ms=4)
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        #  y vs z
        ax = fig.add_subplot(132)
        for element in self.elements:
            ax.plot(element.shape[:, 1], element.shape[:, 2], 'k-')
        for beam in self.beams:
            pos = beam.xyz()
            ax.plot(pos[:, 1], pos[:, 2], 'b-o', lw=2, ms=4)
        ax.set_xlabel('y')
        ax.set_ylabel('z')

        #  x vs y
        ax = fig.add_subplot(133)
        for element in self.elements:
            ax.plot(element.shape[:, 0], element.shape[:, 1], 'k-')
        for beam in self.beams:
            pos = beam.xyz()
            ax.plot(pos[:, 0], pos[:, 1], 'b-o', lw=2, ms=4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

    def plot_detector(self, axes=None):
        """Plot beam positions on final optical element"""
        self.elements[-1].plot_detector_image(axes)

    def beam_distances(self):
        """Calculate beam path distances"""
        dist = np.zeros(len(self.beams))
        for n, beam in enumerate(self.beams):
            dist[n] = beam.total_distance()
        return dist
