"""
Example RayTracer script
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

import beamline_ray_tracer
from beamline_ray_tracer import components

beamline_ray_tracer.fg.nice_print()

#print(beamline_ray_tracer.ft.plane_vectors((0,0,0.1)))

# Instantiate Optical Table (room)
room = beamline_ray_tracer.Room('I16')
room.generate_beams((0, 0, -2), (0, 0, 1), 0.05, 0.02, n_beams=3)

# Create optical components
mono = components.ChannelCutMono('Si111', [0, 0, -1.5], bragg=20, monogap=0.1, length=0.5, width=0.5)
mirror1 = components.CurvedMirrorVertical('CyclindricalMirror', [0, 0, -0.5], [1, 0, 0], radius=2, n_elements=21, length=0.6, width=0.05)
mirror2 = components.CurvedMirrorHorizontal('Bender', [0.37, 0, 0.5], [-1, 0, 0], radius=12, n_elements=21, length=0.6, width=0.1)
slits = components.Slits('Slits', [0, 0, -1], [0, 0, 1], horizontal_gap=0.4, vertical_gap=0.05)
sample = components.Sample('Test', [.38, 0, 1.9], 20, 90, 0, 0, 0.2, 0.2)
detector = components.DetectorArm('Pilatus', [0.38, 0, 1.9], delta=42, gamma=0, distance=1, length=0.4, width=0.1)
kbmirror = components.KBMirror('M', (0, 0, 1), (0, 0, 1), pitch=10, radius=20, n_elements=21, length=0.5, width=0.3)

#el = detector.elements[0]
#el.plot_element()
#plt.show()

#det = el.relative_position(el.shape)
#print(det)
#pos = el.relative_position(np.array(el.store_interactions))
#print(pos)

# Rotate optical components
mono.rotate_bragg(40)
mirror1.rotate(10)
mirror2.rotate(10)

# Add optical components to room (the order doesn't matter)
#room.add_elements(elements)
#room.add_component(mono)
#room.add_component(mirror1)
#room.add_component(mirror2)
room.add_component(kbmirror)
#room.add_component(slits)
#room.add_component(sample)
# Detector must go last
#room.add_component(detector)
room.add_absorber('Detector', [0, 0, 2], [0, 0, -1], 2, 1)

print(room.elements[-1].shape)
print(room.elements[-1].vectors)
print(room)

# Run the ray-tracing simulations
room.run(debug=True)

# Plot the results
room.plot()  # 3D plot
room.plot_projections()  # 2D plot
room.plot_detector()  # Plot beam incidence on final component
plt.show()

print('Distances')
distances = room.beam_distances()
for n, beam in enumerate(room.beams):
    print(beam, beam.total_distance())
