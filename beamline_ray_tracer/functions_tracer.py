"""
beamline_ray_tracer functions
"""

import numpy as np


def reflection(beam_direction, mirror_normal):
    """
    Return reflected vector
    """
    return -2 * np.dot(beam_direction, mirror_normal) * mirror_normal + beam_direction


def plane_vectors(normal, length=1.0, width=1.0, horizontal_direction=None):
    """
    Return plane [a,b,c] where:
        c is the normal (unit) vector
        b is the projection along [0,0,1] (or [1,0,0] if parallel to [0,0,1])
        a is perpendicular to c and b

    :param normal: [dx,dy,dz] normal vector (will be normalised)
    :param length: float, length of vector a (vertical)
    :param width: float, length of vector b (horizontal)
    :param horizontal_direction: [dx,dy,dz], b will be the projection along this vector
    :returns: array([a, b, c]) = vertical, horizontal, normal
    """
    normal = np.asarray(normal, dtype=np.float) / np.sqrt(np.sum(np.square(normal)))

    # Default length is along the z direction
    if horizontal_direction is None:
        hdir = [0, 0, 1]
    else:
        hdir = np.asarray(horizontal_direction, dtype=np.float) / np.sqrt(np.sum(np.square(horizontal_direction)))
    if np.abs(np.dot(hdir, normal)) > 0.9:
        if np.abs(np.dot(hdir, [0, 0, 1])) > 0.9:
            hdir = [1, 0, 0] # if normal is along (001), b(width) || (010), a(height) || (100)
        else:
            hdir = [0, 0, 1] # otherwise, b(width) = (001) X normal e.g. (010)
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

    [len_vec, wid_vec, normal] = plane_vectors(normal, horizontal_direction=None)
    d1 = wid_vec * width / 2
    d2 = len_vec * length / 2
    d3 = -wid_vec * width / 2
    d4 = -len_vec * length / 2

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


def sagittal_focal_length(pitch, radius=0.1):
    """
    Return the focal distance of a vertically focussing cyclindrical mirror
    :param pitch: float : mirror pitch in degrees
    :param radius: float : mirror radius in m
    :returns: float : focal distance in m
    """
    return radius / (2 * np.sin(np.deg2rad(pitch)))


def meridonal_focal_length(pitch, radius=1000):
    """
    Return the focal distance of a horizontally focussing curved mirror
    :param pitch: float : mirror pitch in degrees
    :param radius: float : mirror radius in m
    :returns: float : focal distance in m
    """
    return 0.5 * radius * np.sin(np.deg2rad(pitch))


def mirror_reflection2d(mirror_pos, mirror_dir, beam_pos, beam_dir, radius):
    """
    Calculate the intercept of a beam on a circular mirror in 2D
      mirror_pos defines the position of the mirror plane,
        where the centre of the mirror curvature is.
      mirror_dir defines the mirror plane normal, pointing away from the mirror origin.

    :returns mirror_intercept: [x, y] location of mirror intercept
    :returns relative_intercpt: [a, b] intercept relative to mirror centre, a is along mirror plane, b along normal vector
    :returns reflected beam: [dx, dy] direction of reflected beam
    """
    from .functions_general import vector_intersection
    def c(deg): return np.cos(np.deg2rad(deg))
    def mag(vec): return np.sqrt(np.sum(np.square(vec)))

    # Mirror is defined by a flat plane, distance radius from mirror origin
    m_pos = np.array(mirror_pos)
    m_dir = np.array(mirror_dir) / mag(mirror_dir)
    mirror_origin = m_pos - radius * m_dir
    plane_dir = np.array([-m_dir[1], m_dir[0]])  # rotation by 90deg

    vec_pos = np.array(beam_pos)
    vec_dir = np.array(beam_dir) / mag(beam_dir)

    # Intersection of beam vector and plane perpendicular to mirror normal
    # 2D only!
    intersect = vector_intersection(m_pos, plane_dir, vec_pos, vec_dir)

    # Vector from mirror origin to plane-intercept
    op = intersect - mirror_origin
    opmag = mag(op)  # distance from plane intercept to origin
    opang = np.rad2deg(np.arccos(np.dot(op / opmag, vec_dir)))  # angle difference between plane-intercept and vector

    # Use cosine rule R^2 = y^2 + opmag**2 + 2.y.opmag.cos(opang)
    # solve for y - distance from plane intersept to mirror intersept
    # using quadratic formula: y=(-b +/- sqrt(b^2-4ac))/2a
    # forumal has 2 solutions, try both and take the closest to plane intersept.
    y1 = (2 * opmag * c(opang) + np.sqrt((2 * opmag * c(opang))**2 - 4 * (opmag**2 - radius**2)))/2
    y2 = (2 * opmag * c(opang) - np.sqrt((2 * opmag * c(opang))**2 - 4 * (opmag**2 - radius**2)))/2
    y = min(y1, y2)
    mirror_intersect = intersect - y * vec_dir

    # Find the relative positions to mirror intercept from mirror position.
    x_true = np.dot(mirror_intersect - m_pos, plane_dir)
    phi_true = np.rad2deg(np.arcsin(x_true / radius))
    z_true = radius - radius * c(phi_true)  # distance from plane to arc
    rel_int = [x_true, z_true]

    # Find the reflected vector from the mirror intercept
    om = mirror_intersect - mirror_origin
    ommag = mag(om)  # distance from intercept to origin - should be radius
    ref = reflection(beam_direction=vec_dir, mirror_normal=om / ommag)
    return mirror_intersect, rel_int, ref


def mirror_intercept(vector, plane_intersept, mirror_origin, radius):
    """
    Calculate the intercept of a beam on a circular mirror positioned on flat plane
    """
    def c(deg): return np.cos(np.deg2rad(deg))
    def mag(vec): return np.sqrt(np.sum(np.square(vec)))

    vector = np.asarray(vector)
    plane_intersept = np.asarray(plane_intersept)

    # Vector from mirror origin to plane-intercept
    op = plane_intersept - mirror_origin
    opmag = mag(op)  # distance from plane intercept to origin
    opang = np.rad2deg(np.arccos(np.dot(op / opmag, vector)))  # angle difference between plane-intercept and vector

    # Use cosine rule R^2 = y^2 + opmag**2 + 2.y.opmag.cos(opang)
    # solve for y - distance from plane intersept to mirror intersept
    # using quadratic formula: y=(-b +/- sqrt(b^2-4ac))/2a
    # forumal has 2 solutions, try both and take the closest to plane intersept.
    y1 = (2 * opmag * c(opang) + np.sqrt((2 * opmag * c(opang)) ** 2 - 4 * (opmag ** 2 - radius ** 2))) / 2
    y2 = (2 * opmag * c(opang) - np.sqrt((2 * opmag * c(opang)) ** 2 - 4 * (opmag ** 2 - radius ** 2))) / 2
    y = min(y1, y2)
    mirror_intersect = plane_intersept - y * vector
    return mirror_intersect


def cylinder_intercept(vector, plane_intersept, mirror_origin, radius, mirror_normal, mirror_vertical):
    """
    Calculate the intercept of a beam on a circular mirror positioned on flat plane
    """
    def c(deg): return np.cos(np.deg2rad(deg))
    def mag(vec): return np.sqrt(np.sum(np.square(vec)))

    vector = np.asarray(vector) / mag(vector) # unit vector
    plane_intersept = np.asarray(plane_intersept)
    mirror_normal = np.asarray(mirror_normal) / mag(mirror_normal)  # unit vector
    mirror_vertical = np.asarray(mirror_vertical) / mag(mirror_vertical) # unit vector, perp. to normal
    mirror_horizontal = np.cross(mirror_normal, mirror_vertical)
    # print('    mirror_normal: %s' % mirror_normal)
    # print('  mirror_vertical: %s' % mirror_vertical)
    # print('mirror_horizontal: %s' % mirror_horizontal)

    # tranform the intersept, origin and vector into mirror-space
    u_vectors = [mirror_vertical, mirror_horizontal, mirror_normal]
    [i_intersect, i_origin, i_vec] = np.dot([plane_intersept, mirror_origin, vector], np.linalg.inv(u_vectors))
    # print('Initial, indexed on mirror plane:')
    # print('  plane_intersept: %s, %s' % (plane_intersept, i_intersect))
    # print('    mirror_origin: %s, %s' % (mirror_origin, i_origin))
    # print('           vector: %s, %s' % (vector, i_vec))

    # Vector from mirror origin to plane-intercept
    op = i_intersect[[0, 2]] - i_origin[[0, 2]]
    ii_vec = i_vec[[0, 2]] / mag(i_vec[[0, 2]])
    opmag = mag(op)  # distance from plane intercept to origin
    opang = np.rad2deg(np.arccos(np.dot(op / opmag, ii_vec)))  # angle difference between plane-intercept and vector

    # Use cosine rule R^2 = y^2 + opmag**2 + 2.y.opmag.cos(opang)
    # solve for y - distance from plane intersept to mirror intersept
    # using quadratic formula: y=(-b +/- sqrt(b^2-4ac))/2a
    # forumal has 2 solutions, try both and take the closest to plane intersept.
    y1 = (2 * opmag * c(opang) + np.sqrt((2 * opmag * c(opang)) ** 2 - 4 * (opmag ** 2 - radius ** 2))) / 2
    y2 = (2 * opmag * c(opang) - np.sqrt((2 * opmag * c(opang)) ** 2 - 4 * (opmag ** 2 - radius ** 2))) / 2
    y = min(abs(y1), abs(y2))

    # print('opang: %.2f\n   y1: %.2f, y2: %.2f' % (opang, y1, y2))

    im_intersect = np.array([
        i_intersect[0] - y * ii_vec[0],
        i_intersect[1] - y * i_vec[1] / mag(i_vec[[0, 2]]),
        i_intersect[2] - y * ii_vec[1],
    ])
    mirror_intersect = np.dot(im_intersect, u_vectors)
    # print('     im_intersect: %s\n mirror_intersect: %s' % (im_intersect, mirror_intersect))

    im_normal = np.array([
        im_intersect[0] - i_origin[0],  # vertical (arc)
        0, #- i_origin[1],  # horizontal (cylinder vertical)
        im_intersect[2] - i_origin[2]  # normal
    ])
    #im_normal = im_intersect - i_origin
    intersect_normal = np.dot(im_normal / mag(im_normal), u_vectors)
    # print('        im_normal: %s\n intersect_normal: %s' % (im_normal, intersect_normal))
    return mirror_intersect, intersect_normal
