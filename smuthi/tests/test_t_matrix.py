# -*- coding: utf-8 -*-
"""Test the functions defined in t_matrix.py."""

import unittest
import smuthi.t_matrix
import smuthi.index_conversion

class MieTest(unittest.TestCase):
    def setUp(self):
        self.tau = 0
        self.l = 4
        self.m = -2
        self.omega = 2 * 3.15 / 550
        self.n_medium = 1.6 + 0.1j
        self.n_particle = 2.6 + 0.4j
        self.radius = 260
        self.lmax = 5

    def test_Mie_against_prototype(self):
        q = smuthi.t_matrix.mie_coefficient(0, self.l, self.omega * self.n_medium, self.omega * self.n_particle,
                                            self.radius)
        self.assertAlmostEqual(q, -0.421469104177215 - 0.346698396584972j)
        q = smuthi.t_matrix.mie_coefficient(1, self.l, self.omega * self.n_medium, self.omega * self.n_particle,
                                            self.radius)
        self.assertAlmostEqual(q, -0.511272170262304 - 0.061284547954858j)

    def test_tmatrix(self):
        t = smuthi.t_matrix.t_matrix_sphere(self.omega * self.n_medium, self.omega * self.n_particle, self.radius,
                                            self.lmax)
        n = smuthi.index_conversion.multi2single(self.tau, self.l, self.m, self.lmax)
        mie = smuthi.t_matrix.mie_coefficient(self.tau, self.l, self.omega * self.n_medium,
                                              self.omega * self.n_particle, self.radius)
        self.assertEqual(t[n, n], mie)



if __name__ == '__main__':
    unittest.main()
