"""
Example beamline_ray_tracer script
Create a room container and add some example components, then run and plot
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

import beamline_ray_tracer
from beamline_ray_tracer import fg, ft
from beamline_ray_tracer import components
from beamline_ray_tracer import elements

beamline_ray_tracer.fg.nice_print()

# Create Room
room = beamline_ray_tracer.Room('Two Curved Mirrors')
# Generate set of beams orginating at "position" in direction of "direction"
room.generate_beams(
    position=(0,0,-2),
    direction=(0,0,1),
    horizontal_width=0.005,
    vertical_width=0.005,
    n_beams=5
)

mirror_sep = 2
mirror_pitch = 0.228
m1_radius = 0.096
m2_radius = 5200


# Mirrors
mirrors = components.MirrorSystem(
    name='MainMirrors',
    position=[0, 0, 0],
    pitch=mirror_pitch,
    m1m2distance=mirror_sep,
    m1_radius=m1_radius,
    m2_radius=m2_radius,
    n_elements=11,
    length=2,
    width=0.03
)

# The mirrors focus a long way from the mirror positon, add a detector
bpm = components.Absorber('BPM', [0, 0, 12], [0, 0, 1], length=0.05, width=0.01)

# Add mirror to room
room.add_component(mirrors)
room.add_element_list([bpm])

# Run ray tracing simulation
room.extrude_length(0.)
room.run()

#print(room)

# Plot the results
room.plot()  # 3D plot

room.plot_projections(False)

mirrors.plot_projections()

room.plot_detector()
