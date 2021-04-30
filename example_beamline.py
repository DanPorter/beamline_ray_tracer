"""
beamline_ray_tracer example
Full Beamline Example with several elements and reaslistic distances
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import beamline_ray_tracer
from beamline_ray_tracer import components
from beamline_ray_tracer.classes_diffractometer import Diffractometer

beamline_ray_tracer.fg.nice_print()


def cal_bragg(energy_kev):
    si_latt = 5.431020511  # Silicon
    d_space = si_latt / np.sqrt(3)  # (111)
    # n*lambda = 2d sin theta
    return np.rad2deg(np.arcsin(6.19922 / (energy_kev * d_space)))


def beamwidth(detector):
    widx, widy = detector.beam_width()
    print('Width %s X: %1.3f mm, Y: %1.3f mm' % (detector.name, widx * 1000, widy * 1000))


def beamsize(pitch, sagittal_radius, meridonal_radius, d_sample, d_mirrors, max_vert, max_horiz):
    """
    Calculate beam size
    """
    sinth = np.sin(np.deg2rad(pitch))
    s_v = np.abs(max_vert * (1 - (2 * d_sample * sinth)/sagittal_radius))
    s_h = np.abs(max_horiz * (1 - 2 * (d_sample - d_mirrors) / (meridonal_radius * sinth)))
    return s_v, s_h


# All distances in m, angles in degrees
energy = 8  # keV
bragg = cal_bragg(energy)
monogap = 0.007
mirror_distance = 12  # distance from mirrors to sample
mirror_sep = 2
mirror_pitch = 0.228
m1_radius = 0.096
m2_radius = 5200

beam_height = 2 * monogap * np.cos(np.deg2rad(bragg))
beam_deflection = mirror_sep * np.tan(2 * np.deg2rad(mirror_pitch))
vertical_focus = m1_radius / (2 * np.sin(np.deg2rad(mirror_pitch)))
horizontal_focus = 0.5 * m2_radius * np.sin(np.deg2rad(mirror_pitch))
beamsize_vert, beamsize_horiz = beamsize(
    pitch=mirror_pitch,
    sagittal_radius=m1_radius,
    meridonal_radius=m2_radius,
    d_sample=mirror_distance,
    d_mirrors=mirror_sep,
    max_vert=0.005,
    max_horiz=0.005
)

print('Energy: %5.3f keV, Bragg = %5.3f Deg' % (energy, bragg))
print('Beam Height: %1.3f m' % beam_height)
print('Mirror pitch: %1.3f Deg (%1.2f mrad), bender: %1.0f m' % (mirror_pitch, 1000*np.deg2rad(mirror_pitch), m2_radius))
print(' Beam Deflection: %1.3f m' % beam_deflection)
print('  Vertical focus: %1.3f m' % vertical_focus)
print('Horizontal focus: %1.3f m' % horizontal_focus)
print('Predicted beam size at sample: X: %1.3f mm, Y: % 1.3f mm' % (beamsize_horiz * 1000, beamsize_vert * 1000))

# Instantiate Optical Table (room)
room = beamline_ray_tracer.Room('I16')
room.generate_beams((-beam_deflection, -beam_height, -mirror_distance-3), (0, 0, 1), 0.005, 0.005, n_beams=7)

" Create optical components "
# Mono
mono = components.ChannelCutMono('Si111', [-beam_deflection, -beam_height, -mirror_distance-2], bragg=bragg,
                                 monogap=monogap, length=0.05, width=0.03)
# Mirrors
mirrors = components.MirrorSystem('MainMirrors', [-beam_deflection, 0, -mirror_distance], mirror_pitch,
                                  m1m2distance=mirror_sep, m1_radius=m1_radius, m2_radius=m2_radius,
                                  n_elements=101, length=2, width=0.03)
# BPM
bpm = components.Window('BPM', [0, 0, -2], [0, 0, 1], length=0.05, width=0.05)
bpm_sample = components.Window('Sample', [0, 0, -0.1], [0, 0, 1], length=0.05, width=0.05)
# Sample slits
slits = components.Slits('Slits', [0, 0, -1], [0, 0, 1], horizontal_gap=0.01, vertical_gap=0.01, plate_width=0.05)
# Diffractometer
diffractometer = Diffractometer('Pilatus', [0, 0, 0], eta=20, chi=90, phi=0, mu=0, delta=40, gamma=0,
                                detector_distance=1, sample_length=0.1, sample_width=0.2, detector_normal=(0, 0, 1),
                                detector_pixels=(487, 195), pixel_size=172e-6)
# Big End of Room Detector (catch beams that don't hit)
catch = components.Detector('catch', [0, 0, 2], [0, 0, 1], size=5)


# Add optical components to room (the order doesn't matter)
room.add_component(mono)
room.add_component(mirrors)
room.add_element_list([bpm, bpm_sample])
room.add_component(slits)
room.add_component(diffractometer)
#room.add_component(catch)

# Run the ray-tracing simulations
room.run()

# Plot the results
room.plot()  # 3D plot
room.plot_projections(False)  # 2D plot
room.plot_detector()  # Plot beam incidence on final component

# print beamsize at sample
beamwidth(bpm_sample)

plt.figure(figsize=[15, 8], dpi=60)
ax1 = plt.subplot(121)
bpm.plot_detector_image(ax1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('bpm\n%s' % bpm.beam_str())
ax1.axis('image')
ax2 = plt.subplot(122)
bpm_sample.plot_detector_image(ax2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Sample\n%s' % bpm_sample.beam_str())
ax2.axis('image')
plt.ioff()
plt.show()



