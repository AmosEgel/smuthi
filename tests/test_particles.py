# -*- coding: utf-8 -*-
"""Test the ParticleCollection class"""

import smuthi.particles


def test_adding_particles():
    pcln = smuthi.particles.ParticleCollection()

    pcln.add_sphere(radius=200, refractive_index=3 + 2j, position=[200.55, 400.11, -100.33])
    pcln.add_sphere(radius=200, refractive_index=3 + 2j, position=[200.55, 400.11, -100.33])
    pcln.add_sphere(radius=200.01, refractive_index=3 + 2j, position=[700.55, 200.11, 100.33])

    pcln.add_spheroid(semi_axis_a=100, semi_axis_c=200, refractive_index=1.1, position=[500, 200, 300],
                      euler_angles=[0, 2, 3])
    pcln.add_spheroid(semi_axis_a=100, semi_axis_c=200, refractive_index=1.1, position=[100, 100, 100],
                      euler_angles=[0, 2, 3])
    pcln.add_spheroid(semi_axis_a=300, semi_axis_c=100, refractive_index=1.1, position=[-100, -100, 100],
                      euler_angles=[0, 2, 3])

    pcln.add_finite_cylinder(cylinder_height=200, cylinder_radius=100, refractive_index=3 + 1j,
                             position=[-1000, 1000, 2000])
    pcln.add_finite_cylinder(cylinder_height=200, cylinder_radius=100, refractive_index=3 + 1j,
                             position=[1000, -1000, 2000])
    pcln.add_finite_cylinder(cylinder_height=300, cylinder_radius=100, refractive_index=3 + 1j,
                             position=[1000, -1000, -2000])

    assert pcln.particle_number() == 9
    assert pcln.particle_positions()[4] == [100, 100, 100]

    pcln.remove_particle(3)
    assert pcln.particle_positions()[4] == [-100, -100, 100]


if __name__ == '__main__':
    test_adding_particles()
