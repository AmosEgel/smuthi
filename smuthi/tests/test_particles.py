# -*- coding: utf-8 -*-
"""Test the ParticleCollection class"""

import unittest
import particles


class ParticleCollectionTest(unittest.TestCase):
    def test_adding_particles(self):
        pcln = particles.ParticleCollection()

        pcln.add_sphere(radius=200, refractive_index=3 + 2j, x=200.55, y=400.11, z=-100.33)
        pcln.add_sphere(radius=200, refractive_index=3 + 2j, x=200.55, y=-400.11, z=-100.33)
        pcln.add_sphere(radius=200.01, refractive_index=3 + 2j, x=700.55, y=200.11, z=100.33)

        pcln.add_spheroid(semi_axis_a=100, semi_axis_c=200, refractive_index=1.1, x=500, y=200, z=300, alpha=0, beta=2,
                          gamma=3)
        pcln.add_spheroid(semi_axis_a=100, semi_axis_c=200, refractive_index=1.1, x=100, y=100, z=100, alpha=0, beta=2,
                          gamma=3)
        pcln.add_spheroid(semi_axis_a=300, semi_axis_c=100, refractive_index=1.1, x=-100, y=-100, z=100, alpha=0,
                          beta=2, gamma=3)

        pcln.add_finite_cylinder(cylinder_height=200, cylinder_radius=100, refractive_index=3 + 1j, x=-1000, y=1000,
                                 z=2000)
        pcln.add_finite_cylinder(cylinder_height=200, cylinder_radius=100, refractive_index=3 + 1j, x=1000, y=-1000,
                                 z=2000)
        pcln.add_finite_cylinder(cylinder_height=300, cylinder_radius=100, refractive_index=3 + 1j, x=1000, y=-1000,
                                 z=-2000)

        self.assertEqual(pcln.particle_number(), 9)
        self.assertEqual(len(pcln.specs_list), 6)
        self.assertEqual(pcln.specs_indices, [0, 0, 1, 2, 2, 3, 4, 4, 5])
        self.assertEqual(pcln.positions[4], [100, 100, 100])

        pcln.remove_particle(3)
        self.assertEqual(pcln.positions[4], [-100, -100, 100])
        self.assertEqual(pcln.specs_indices, [0, 0, 1, 2, 3, 4, 4, 5])


if __name__ == '__main__':
    unittest.main()
