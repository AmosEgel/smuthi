# -*- coding: utf-8 -*-
"""Provide class for the representation of scattering particles."""


class Particle:
    """Base class for scattering particles."""
    def __init__(self, position=[0,0,0], euler_angles=[0,0,0], refractive_index=1+0j):
        self.position = position
        self.refractive_index = refractive_index


class Sphere(Particle):
    def __init__(self, position=[0,0,0], refractive_index=1+0j, radius=1):
        Particle.__init__(self, position=position, refractive_index=refractive_index)
        self.radius = radius


class Spheroid(Particle):
    def __init__(self, position=[0,0,0], euler_angles=[0,0,0], refractive_index=1+0j, semi_axis_c=1, semi_axis_a=1):
        Particle.__init__(self, position=position, euler_angles=euler_angles, refractive_index=refractive_index)
        self.semi_axis_c = semi_axis_c
        self.semi_axis_a = semi_axis_a


class FiniteCylinder(Particle):
    def __init__(self, position=[0,0,0], euler_angles=[0,0,0], refractive_index=1+0j, cylinder_radius=1,
                 cylinder_height=1):
        Particle.__init__(self, position=position, euler_angles=euler_angles, refractive_index=refractive_index)
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height


class ParticleCollection:
    """Collection of scattering particles."""
    def __init__(self, particle_list=None):
        """A list of dictionaries Particle objects"""
        if particle_list is None:
            self.particles = []
        else:
            self.particles = particle_list

    def add(self, particle):
        """Add particle to collection"""
        self.particles.append(particle)

    def remove_particle(self, i):
        """Remove i-th particle from collection"""
        del self.particles[i]

    def particle_number(self):
        """Return total number of particles in collection"""
        return len(self.particles)

    def particle_positions(self):
        """Return a list of particle positions"""
        return [self.particles[i].position for i in range(self.particle_number())]
