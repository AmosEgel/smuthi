# -*- coding: utf-8 -*-
"""Test the functions defined in t_matrix.py."""

import smuthi.linearsystem.tmatrix.t_matrix as tmt
import smuthi.particles
import smuthi.fields.expansions as fldex
import numpy as np

tau = 0
l = 4
m = -2
omega = 2 * 3.15 / 550
n_medium = 1.6 + 0.1j
n_particle = 2.6 + 0.4j
radius = 260
lmax = 10
mmax = 10


def test_Mie_against_prototype():
    q = tmt.mie_coefficient(0, l, omega * n_medium, omega * n_particle, radius)
    np.testing.assert_almost_equal(q, -0.421469104177215 - 0.346698396584972j)
    q = tmt.mie_coefficient(1, l, omega * n_medium, omega * n_particle, radius)
    np.testing.assert_almost_equal(q, -0.511272170262304 - 0.061284547954858j)


def test_tmatrix():
    t = tmt.t_matrix_sphere(omega * n_medium, omega * n_particle, radius, lmax, mmax)
    n = fldex.multi_to_single_index(tau, l, m, lmax, mmax)
    mie = tmt.mie_coefficient(tau, l, omega * n_medium, omega * n_particle, radius)
    assert t[n, n] == mie

    sphere1 = smuthi.particles.Sphere(radius=100, refractive_index=3, position=[100, 200, 300], l_max=lmax, m_max=mmax)
    sphere2 = smuthi.particles.Sphere(radius=100, refractive_index=3, position=[200, -200, 200], l_max=lmax, m_max=mmax)
    sphere3 = smuthi.particles.Sphere(radius=200, refractive_index=2+0.2j, position=[200,-200,200], l_max=lmax,
                                      m_max=mmax)

    t2 = tmt.t_matrix(vacuum_wavelength=550, n_medium=n_medium, particle=sphere1)
    t3 = tmt.t_matrix_sphere(2*np.pi/550 * n_medium, 2*np.pi/550 * 3, 100, lmax, mmax)
    np.testing.assert_allclose(t2, t3)


if __name__ == '__main__':
    test_Mie_against_prototype()
    test_tmatrix()
