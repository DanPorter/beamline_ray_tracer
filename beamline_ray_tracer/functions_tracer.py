"""
beamline_ray_tracer functions
"""

import numpy as np

from . import functions_general as fg


def reflection(beam_direction, mirror_normal):
    """
    Return reflected vector
    """
    return -2 * np.dot(beam_direction, mirror_normal) * mirror_normal + beam_direction


def plane_vectors(normal, length=1.0, width=1.0, horizontal_direction=None):
    """
    Return plane [a,b,c] where:
        c is the normal vector
        a is the projection along [0,0,1] (or [1,0,0] if parallel to [0,0,1])
        b is perpendicular to c and a
    """
    normal = np.asarray(normal, dtype=np.float) / np.sqrt(np.sum(np.square(normal)))

    # Default length is along the z direction
    if horizontal_direction is None and np.abs(np.dot([0, 0, 1], normal)) > 0.9:
        b = np.cross([1, 0, 0], normal)
    elif horizontal_direction is None:
        b = np.cross([0, 0, 1], normal)
    else:
        hdir = np.asarray(horizontal_direction, dtype=np.float) / np.sqrt(np.sum(np.square(horizontal_direction)))
        b = np.cross(hdir, normal)
    a = np.cross(normal, b)

    a = length * a / np.sqrt(np.sum(np.square(a)))
    b = width * b / np.sqrt(np.sum(np.square(b)))
    return np.array([a, b, normal])


def plane_points(centre, normal, length=1.0, width=1.0, horizontal_direction=None):
    """
    Return square points on a plane for plotting
    """
    centre = np.asarray(centre, dtype=np.float)
    normal = np.asarray(normal, dtype=np.float) / np.sqrt(np.sum(np.square(normal)))

    # Default length is along the z direction
    if horizontal_direction is None and np.abs(np.dot([0, 0, 1], normal)) > 0.9:
        d1 = np.cross([1, 0, 0], normal)
    elif horizontal_direction is None:
        d1 = np.cross([0, 0, 1], normal)
    else:
        hdir = np.asarray(horizontal_direction, dtype=np.float) / np.sqrt(np.sum(np.square(horizontal_direction)))
        d1 = np.cross(hdir, normal)
    d2 = np.cross(d1, normal)
    d3 = np.cross(d2, normal)
    d4 = np.cross(d3, normal)

    d1 = d1 / np.sqrt(np.sum(np.square(d1)))
    d2 = d2 / np.sqrt(np.sum(np.square(d2)))
    d3 = d3 / np.sqrt(np.sum(np.square(d3)))
    d4 = d4 / np.sqrt(np.sum(np.square(d4)))

    d1 = d1 * width / 2
    d2 = d2 * length / 2
    d3 = d3 * width / 2
    d4 = d4 * length / 2

    p1 = (d1 + d2) + centre
    p2 = (d2 + d3) + centre
    p3 = (d3 + d4) + centre
    p4 = (d4 + d1) + centre
    return np.array([p1, p2, p3, p4, p1])


def curved_mirror(position, direction, radius=1.0, length=1.0, n_elements=31, horizontal=True):
    """
    Define points on a curved mirror (curved in 1 dimension)
    :param position: [x,y,z] position of mirror centre (back of curve)
    :param direction: [x,y,z] direction of mirror normal (curve facing towards)
    :param radius: float : mirror radius
    :param length: float : mirror height, must be < radius
    :param n_elements: int : number of points to define
    :param horizontal: False/True : if True, the curvature is created along the other direction
    :return: array([[x,y,z]]), array([[dx,dy,dz]])
    """
    position = np.array(position, dtype=np.float)  # Mirror centre (back of curve)
    max_ang = np.arcsin(0.5 * length / radius)
    phi = np.linspace(-max_ang, max_ang, n_elements)

    direction = -np.asarray(direction, dtype=np.float) / np.sqrt(np.sum(np.square(direction)))
    u1, u2, u3 = plane_vectors(direction)
    if horizontal:
        plane = u1
    else:
        plane = u2

    xyz = np.zeros([len(phi), 3])
    dxdydz = np.zeros([len(phi), 3])
    for n, ph in enumerate(phi):
        new_dir = radius * (np.sin(ph) * plane + np.cos(ph) * u3)
        new_pos = new_dir - radius * direction + position
        xyz[n] = new_pos
        dxdydz[n] = -new_dir
    return xyz, dxdydz
