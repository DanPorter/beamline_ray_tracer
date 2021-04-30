# beamline_ray_tracer
Simple ray tracing program to trace reflections of beams from various optical elements.

Originally built to trace beamline optics on a synchrotron beamline, with simple components such as mirrors included.

**Requires:** _numpy, matplotlib_

By Dan Porter, PhD

Diamond Light Source Ltd

2020

### Usage:
```python
from beamline_ray_tracer import Room, components

# Create room (container for beams and optical elements)
room = Room('Optical Table')
room.generate_beams(n_beams=7)
# Define optical components
mirror1 = components.CurvedMirrorVertical('CyclindricalMirror', [0, 0, 1], [-1, 0, -1], radius=3, n_elements=3, length=1, width=1)
mirror2 = components.CurvedMirrorHorizontal('Bender', [-1, 0, 1], [1, 0, 1], radius=3, n_elements=3, length=1, width=1)
# Add components to room
room.add_component(mirror1)
room.add_component(mirror2)
room.add_absorber('Detector', [-1, 0, 2], [0, 0, -1], 2, 1)

# Run the ray-tracing simulations
room.run()
room.plot()
room.plot_projections()
room.plot_detector()
```

![3D Ray Tracing example](./example_3d.svg "Example room.plot()")
![2D Ray Tracing example](./example_projection.svg "Example room.plot_projections()")


