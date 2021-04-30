"""
beamline_ray_tracer

Simple ray tracing program to trace reflections of beams from various optical elements.

Originally built to trace beamline optics on a synchrotron beamline, with simple components such as mirrors included.

Requires: numpy, matplotlib

Usage:
As python script:
 from beamline_ray_tracer import Room, components
 # Create room (container for beams and optical elements)
 room = Room('Optical Table')
 room.generate_beams(n_beams=5)
 # Define optical components
 mirror1 = components.CurvedMirrorVertical('CyclindricalMirror', [0, 0, 0.5], [1, 0, 0], radius=2, n_elements=21, length=0.6, width=0.05)
 mirror2 = components.CurvedMirrorHorizontal('Bender', [0.37, 0, 1.5], [-1, 0, 0], radius=12, n_elements=21, length=0.6, width=0.1)
 # Add components to room
 room.add_component(mirror1)
 room.add_component(mirror2)
 room.add_absorber('Detector', [0, 0, 2], [0, 0, -1], 2, 1)
 # Run the ray-tracing simulations
 room.run()
 room.plot()
 plt.show()

By Dan Porter, PhD
Diamond
2020

Version 0.5.0
Last updated: 30/Apr/2021

Version History:
10/10/20 0.1.0  Version History started.
29/10/20 0.2.0  Added diffractometer file and class
30/04/21 0.5.0  Many improvements, CyclindricalMirror element added and tested
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import functions_general as fg
from . import functions_tracer as ft
from . import classes_components as components
from . import classes_elements as elements
from .classes_room import Room
from .classes_beam import Beam
from .classes_diffractometer import Diffractometer

__version__ = '0.5.0'
__date__ = '30/Apr/2021'



