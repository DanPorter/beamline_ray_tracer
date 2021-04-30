"""
beamline_ray_tracer example
Diffractometer Example with interactive figure

Coordinate system:
  Z is along the beam, Y is vertical up (delta plane), X is horizontal, away from ring (gamma plane)
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

import beamline_ray_tracer
from beamline_ray_tracer.classes_diffractometer import Diffractometer

beamline_ray_tracer.fg.nice_print()
#plt.ion()

# Instantiate Optical Table (room)
room = beamline_ray_tracer.Room('I16')
room.generate_beams((0, 0, -2), (0, 0, 1), 0.05, 0.02, n_beams=7)

# Create optical components
diffractometer = Diffractometer('I16', [0, 0, 0], eta=20, chi=90, phi=0, mu=0, delta=40, gamma=0, detector_distance=1,
                                sample_length=0.1, sample_width=0.2, detector_normal=(0, 0, 1),
                                detector_pixels=(487, 195), pixel_size=172e-6)
room.add_component(diffractometer)
room.run()


"------------------------- Interactive Beam Plot --------------------------"

fig = plt.figure(figsize=[12, 10], dpi=60)
ax = fig.add_subplot(221, projection='3d')

xlim = [-1, 1]
ylim = [-1, 1]
zlim = [-1, 1]

ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)

room.plot(ax)

# Detector
axdet = plt.axes([0.6, 0.1, 0.4, 0.8])
room.plot_detector(axdet)
axdet.axis('image')

axsldr1 = plt.axes([0.1, 0.45, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr2 = plt.axes([0.1, 0.37, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr3 = plt.axes([0.1, 0.29, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr4 = plt.axes([0.1, 0.21, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr5 = plt.axes([0.1, 0.13, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr6 = plt.axes([0.1, 0.05, 0.35, 0.06], facecolor='lightgoldenrodyellow')
sldr1 = plt.Slider(axsldr1, 'phi', -180, 180, valinit=diffractometer.phi, valfmt='%5.2f')
sldr2 = plt.Slider(axsldr2, 'chi', -98, 98, valinit=diffractometer.chi, valfmt='%5.2f')
sldr3 = plt.Slider(axsldr3, 'eta', -20, 200, valinit=diffractometer.eta, valfmt='%5.2f')
sldr4 = plt.Slider(axsldr4, 'mu', 0, 100, valinit=diffractometer.mu, valfmt='%5.2f')
sldr5 = plt.Slider(axsldr5, 'delta', 0, 160, valinit=diffractometer.delta, valfmt='%5.2f')
sldr6 = plt.Slider(axsldr6, 'gamma', 0, 160, valinit=diffractometer.gamma, valfmt='%5.2f')


def update_all(val):
    "Update function for pilatus image"

    phi, chi, eta, mu, delta, gamma = sldr1.val, sldr2.val, sldr3.val, sldr4.val, sldr5.val, sldr6.val

    diffractometer.euler(eta=eta, chi=chi, phi=phi, mu=mu, delta=delta, gamma=gamma)
    room.run()

    ax.clear()
    axdet.clear()

    room.plot(ax)
    room.plot_detector(axdet)

    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    plt.draw()
    # fig.canvas.draw()


sldr1.on_changed(update_all)
sldr2.on_changed(update_all)
sldr3.on_changed(update_all)
sldr4.on_changed(update_all)
sldr5.on_changed(update_all)
sldr6.on_changed(update_all)
plt.show()



