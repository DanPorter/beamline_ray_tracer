"""
beamline_ray_tracer classes
 classes of optical Components - collections of optical elements
  - DetectorArm
  - Sample
  - Diffractometer (Sample + Diffractometer)
"""

import numpy as np

from . import functions_general as fg
from . import functions_tracer as ft
from .classes_components import Component
from .classes_elements import Window, Absorber, Reflector


class DetectorArm(Component):
    """
    DetectorArm Component
    A detector (Absorber) element rotating about a sample origin
    """
    def __init__(self, name, sample_position, delta=0, gamma=0, distance=1, length=1, width=1):
        self.delta = delta
        self.gamma = gamma
        self.distance = distance
        direction = fg.you_normal_vector(0, -delta, 90-gamma)
        position = sample_position - distance * direction
        detector = Absorber(name, position, direction, length=length, width=width)
        super().__init__(name, [detector], 'Detector')
        self.rotation_centre = sample_position
        self.rotation_axis = [1, 0, 0]

    def move(self, dxdydz):
        """Move rotation_centre"""
        position = np.asarray(self.rotation_centre, dtype=np.float).reshape(3)
        dxdydz = np.asarray(dxdydz, dtype=np.float).reshape(3)
        new_position = position + dxdydz
        self.rotation_centre = new_position
        for element in self.elements:
            element.move_by(dxdydz)

    def euler(self, delta=None, gamma=None):
        """Change detector position using eulerian angles"""
        if delta:
            self.delta = delta
        if gamma:
            self.gamma = gamma
        direction = fg.you_normal_vector(0, -self.delta, 90 - self.gamma)
        position = self.rotation_centre - self.distance * direction
        for element in self.elements:
            element.move_to(position)
            element.set_normal(direction)

    def inc_euler(self, delta=None, gamma=None):
        """Increase rotation in Eulerian coordinates"""
        if delta:
            self.delta += delta
        if gamma:
            self.gamma += gamma
        direction = fg.you_normal_vector(0, -self.delta, 90 - self.gamma)
        position = self.rotation_centre - self.distance * direction
        for element in self.elements:
            element.move_to(position)
            element.set_normal(direction)


class Sample(Component):
    """
    Sample Component
    A Reflector element rotating about common diffractometer angles
    """
    def __init__(self, name, position, eta=0, chi=90, phi=0, mu=0, length=1, width=1):
        self.eta = eta
        self.chi = chi
        self.phi = phi
        self.mu = mu
        normal = fg.you_normal_vector(eta, chi, mu)
        crs = fg.rotate_about_axis(np.cross(normal, [0, 0, 1]), normal, phi)
        element = Reflector(name, position, normal, length=length, width=width, horizontal_direction=crs)
        super().__init__(name, [element], 'Sample')
        self.rotation_centre = position
        self.rotation_axis = crs

    def move(self, dxdydz):
        """Move centre and rotation_centre"""
        position = np.asarray(self.rotation_centre, dtype=np.float).reshape(3)
        dxdydz = np.asarray(dxdydz, dtype=np.float).reshape(3)
        new_position = position + dxdydz
        self.rotation_centre = new_position
        for element in self.elements:
            element.move_to(new_position)

    def euler(self, eta=None, chi=None, phi=None, mu=None):
        """Define element rotation in Eulerian coordinates"""
        if eta:
            self.eta = eta
        if chi:
            self.chi = chi
        if phi:
            self.phi = phi
        if mu:
            self.mu = mu
        normal = fg.you_normal_vector(self.eta, self.chi, self.mu)
        crs = fg.rotate_about_axis(np.cross(normal, [0, 0, 1]), normal, self.phi)
        self.rotation_axis = crs
        for element in self.elements:
            element.set_normal(normal, crs)

    def inc_euler(self, eta=None, chi=None, phi=None, mu=None):
        """Increase rotation in Eulerian coordinates"""
        if eta:
            self.eta += eta
        if chi:
            self.chi += chi
        if phi:
            self.phi += phi
        if mu:
            self.mu += mu
        normal = fg.you_normal_vector(self.eta, self.chi, self.mu)
        crs = fg.rotate_about_axis(np.cross(normal, [0, 0, 1]), normal, self.phi)
        self.rotation_axis = crs
        for element in self.elements:
            element.set_normal(normal, crs)


class Diffractometer(Component):
    """
    Diffractometer Component
    """
    def __init__(self, name, position, eta=0, chi=90, phi=0, mu=0, delta=0, gamma=0, detector_distance=1,
                 sample_length=1, sample_width=1, detector_length=1, detector_width=1):
        self.eta = eta
        self.chi = chi
        self.phi = phi
        self.mu = mu
        self.delta = delta
        self.gamma = gamma
        self.detector_distance = detector_distance

        self.sample = Sample(name, position, eta, chi, phi, mu, sample_length, sample_width)
        self.detector = DetectorArm(name, position, delta, gamma, detector_distance, detector_length, detector_width)
        elements = self.sample.elements + self.detector.elements
        super().__init__(name, elements, 'Diffractometer')
        self.rotation_centre = position
        self.rotation_axis = [1, 0, 0]

    def move(self, dxdydz):
        """Move centre and rotation_centre"""
        self.sample.move(dxdydz)
        self.detector.move(dxdydz)

    def euler(self, eta=None, chi=None, phi=None, mu=None, delta=None, gamma=None):
        """Define element rotation in Eulerian coordinates"""
        self.sample.euler(eta, chi, phi, mu)
        self.detector.euler(delta, gamma)

    def inc_euler(self, eta=None, chi=None, phi=None, mu=None, delta=None, gamma=None):
        """Increase rotation in Eulerian coordinates"""
        self.sample.inc_euler(eta, chi, phi, mu)
        self.detector.inc_euler(delta, gamma)
