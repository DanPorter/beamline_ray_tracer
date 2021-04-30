"""
Example Curved Mirror
"""

import numpy as np
import matplotlib.pyplot as plt
import beamline_ray_tracer
from beamline_ray_tracer import components

# Create Room
room = beamline_ray_tracer.Room('Curved Mirror')
# Generate set of beams orginating at "position" in direction of "direction"
nbeams = 7

room.generate_beams(
    position=(0,0,-2),
    direction=(0,0,1),
    horizontal_width=1,
    vertical_width=1,
    n_beams=nbeams
)
"""
beam_dir = (0, 0, 1)
for y in np.arange(0, 0.8, 0.1):
    room.add_beam((0, y, -2), beam_dir)
"""
# Create a mirror with position, direction and size
mirror_radius = 2
mirror_elements = 11
mirror = components.CurvedMirrorVertical(
#mirror = components.CurvedMirrorHorizontal(
    name='Mirror',
    position=(0,0,0),
    normal=(1,0,-1),
    radius=mirror_radius,
    n_elements=mirror_elements,
    length=1,
    width=2
)
#mirror.rotate(5)

# Calculate focal position 1/Ob + 1/foc = 1/im
focus = mirror_radius/2
print('Mirror Focus = R/2 = %.2f m' % focus)
room.extrude_length(2 * focus)


# Add mirror to room
room.add_component(mirror)

# Run ray tracing simulation
room.run()

#print(room)
# Beam intersections - find focal point
beamline_ray_tracer.fg.nice_print()
intersect = []
#cen_idx = (nbeams*nbeams//2 + 1)//2
#beam = room.beams[cen_idx]
current_pos, current_dir = mirror.position, mirror.normal
plt.figure()
vec = np.array([current_pos, current_pos + 2 * current_dir])
plt.plot(vec[:, 1], vec[:, 2], 'k-', lw=2)
for beam in room.beams:
    npos, ndir = beam.current()
    if np.dot(current_dir, ndir) > 0.9999: continue
    p1, p2, dist = beamline_ray_tracer.fg.vector_shortest(current_pos, current_dir, npos, ndir)
    if dist < 1e-6 and np.sqrt(np.sum(np.square(p1-current_pos))) > 0.5:
        intersect += [(p1 + p2) / 2]
        vec = np.array([npos, p2, npos - 2 * ndir])
        plt.plot(vec[:, 1], vec[:, 2], '-+', ms=12, label='%r' % beam)
plt.show()

av_focus = np.mean(intersect, axis=0)
df_focus = np.std(intersect, axis=0)
focus = np.sqrt(np.sum(np.square(mirror.position - av_focus)))
error = np.sqrt(np.sum(np.square(df_focus)))
print('Focal distance = %.3f +/- %.3f' % (focus, error))

# Plot the results
room.plot()  # 3D plot

room.plot_projections()

