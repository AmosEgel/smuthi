# -*- coding: utf-8 -*-
"""Test the functions defined in t_matrix.py."""

import smuthi.t_matrix
import smuthi.index_conversion
import smuthi.particles
import numpy as np

tau = 0
l = 4
m = -2
omega = 2 * 3.15 / 550
n_medium = 1.6 + 0.1j
n_particle = 2.6 + 0.4j
radius = 260
smuthi.index_conversion.l_max = 5
smuthi.index_conversion.m_max = 5


def test_Mie_against_prototype():
    q = smuthi.t_matrix.mie_coefficient(0, l, omega * n_medium, omega * n_particle, radius)
    np.testing.assert_almost_equal(q, -0.421469104177215 - 0.346698396584972j)
    q = smuthi.t_matrix.mie_coefficient(1, l, omega * n_medium, omega * n_particle, radius)
    np.testing.assert_almost_equal(q, -0.511272170262304 - 0.061284547954858j)


def test_tmatrix():
    t = smuthi.t_matrix.t_matrix_sphere(omega * n_medium, omega * n_particle, radius)
    n = smuthi.index_conversion.multi_to_single_index(tau, l, m)
    mie = smuthi.t_matrix.mie_coefficient(tau, l, omega * n_medium, omega * n_particle, radius)
    assert t[n, n] == mie

    prtcl = smuthi.particles.ParticleCollection()
    prtcl.add_sphere(100, 3, [100, 200, 300])
    prtcl.add_sphere(100, 3, [200, -200, 200])
    prtcl.add_sphere(200, 2 + 0.2j, [200, -200, 200])

    t2 = smuthi.t_matrix.t_matrix(vacuum_wavelength=550, n_medium=n_medium, particle=prtcl.particles[0])
    t3 = smuthi.t_matrix.t_matrix_sphere(2*np.pi/550 * n_medium, 2*np.pi/550 * 3, 100)
    np.testing.assert_allclose(t2, t3)


if __name__ == '__main__':
    test_Mie_against_prototype()
    test_tmatrix()
