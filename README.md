# beamline_ray_tracer
Simple ray tracing program to trace reflections of beams from various optical elements.

Originally built to trace beamline optics on a synchrotron beamline, with simple components such as mirrors included.

**Requires:** _numpy, matplotlib_

By Dan Porter, PhD

Diamond Light Source Ltd

2020

### Usage:
```python
import beamline_ray_tracer
from beamline_ray_tracer import components
# Create room (container for beams and optical elements)
room = beamline_ray_tracer.Room('Optical Table')
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
```



