# -*- coding: utf-8 -*-
"""Provide class for the representation of scattering particles."""


class ParticleCollection:
    """Collection of scattering particles."""
    def __init__(self):
        """ A list of dictionaries that contain the following entries:
        - 'shape':              'sphere', 'spheroid', 'finite cylinder'
        - 'refractive index'    Complex refractive index in the form n+kj
        - 'position':           In the form [x, y, z] (length unit)
        - 'euler angles':       Euler angles of rotated particle in the format [alpha, beta, gamma] (radian)
        - further shape-specific parameters which characterize the geometry (like radius etc., see adding methods)
        """
        self.particles = []

    def add_sphere(self, radius, refractive_index, position):
        """Add a sphere to the collection.

        input parameters:
        radius:             sphere radius (length unit)
        refractive_index:   complex refractive index in the form n+jk
        position:           center position in format [x,y,z] (length unit)
        """
        self.particles.append({'shape': 'sphere', 'radius': radius, 'refractive index': refractive_index,
                               'position': position, 'euler angles': [0, 0, 0]})

    def add_spheroid(self, semi_axis_c, semi_axis_a, refractive_index, position, euler_angles=[0, 0, 0]):
        """Add a spheroid to the collection.

        input parameters:
        semi_axis_c:        spheroid semi axis along symmetry axis (length unit)
        semi_axis_a:        spheroid semi axis a=b in transverse direction (length unit)
        refractive_index:   complex refractive index in the form n+jk
        position:           center position in format [x,y,z] (length unit)
        euler_angles:       todo, default=[0,0,0]
        """
        self.particles.append({'shape': 'spheroid', 'semi axis c': semi_axis_c, 'semi axis a': semi_axis_a,
                               'refractive index': refractive_index, 'position': position,
                               'euler angles': euler_angles})

    def add_finite_cylinder(self, cylinder_radius, cylinder_height, refractive_index, position, euler_angles=[0, 0, 0]):
        """Add finite cylinder to the collection.

        input parameters:
        cylinder_radius:    cylinder radius (length unit)
        cylinder_height:    height of cylinder (length unit)
        refractive_index:   complex refractive index in the form n+jk
        position:           center position in format [x,y,z] (length unit)
        euler_angles:       todo, default=[0,0,0]
        """
        self.particles.append({'shape': 'finite cylinder', 'cylinder radius': cylinder_radius,
                               'cylinder height': cylinder_height, 'refractive index': refractive_index,
                               'position': position, 'euler angles': euler_angles})

    def remove_particle(self, i):
        """Remove i-th particle from collection"""
        del self.particles[i]

    def particle_number(self):
        """Return total number of particles in collection"""
        return len(self.particles)

    def particle_positions(self):
        """Return a list of particle positions"""
        return [self.particles[i]['position'] for i in range(self.particle_number())]
