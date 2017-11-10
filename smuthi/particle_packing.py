import smuthi.particles as part
import random
import numpy as np
import sys


def random_sequential_addition(zmin, zmax, rhomax=None, volume_density=None, particle_list=None):
    if rhomax is None:
        particle_volume = sum([4 / 3 * np.pi * particle.radius**3 for particle in particle_list])
        cylinder_volume = particle_volume / volume_density
        rhomax = np.sqrt(cylinder_volume / (zmax - zmin) / np.pi)
    for i, particle in enumerate(particle_list):
        sys.stdout.write('\rPacking particle number %i'%i)
        if not type(particle) == part.Sphere:
            raise NotImplementedError('implemented currently only for spheres')
        collision_flag = True
        while collision_flag:
            z = random.uniform(zmin + particle.radius, zmax - particle.radius)
            x, y = np.Inf, np.Inf
            while x**2 + y**2 > (rhomax - particle.radius)**2:
                x = random.uniform(-rhomax + particle.radius, rhomax - particle.radius)
                y = random.uniform(-rhomax + particle.radius, rhomax - particle.radius)
            if i > 0:
                positions_array = np.array([particle.position for particle in particle_list[:i]])
                radii_array = np.array([particle.radius for particle in particle_list[:i]])
                collision_flag = check_sphere_collision(np.array([x, y, z]), particle.radius, positions_array,
                                                        radii_array)
            else:
                collision_flag = False

        particle.position[0] = x
        particle.position[1] = y
        particle.position[2] = z


def check_sphere_collision(new_position, new_radius, old_positions, old_radii):
    squared_distances = ((new_position[None, :] - old_positions)**2).sum(axis=-1)
    return any(squared_distances < (new_radius + old_radii)**2)
