# -*- coding: utf-8 -*-
"""Test the functions defined in vector_wave_functions.py."""

import numpy as np
import smuthi.fields.vector_wave_functions as vwf

tau = 0
l = 4
m = -2
pol = 0
omega = 2 * 3.15 / 550
kp = omega * 1.7
alpha = 3.15/7
kz = np.sqrt(omega**2 - kp**2 + 0j)
x = 120
y = -200
z = 80
z2 = 500
lmax = 20


def test_PVWF_against_prototype():
    Ex, Ey, Ez = vwf.plane_vector_wave_function(x, y, z, kp, alpha, kz, 0)
    
    np.testing.assert_almost_equal(Ex, -0.113172457895318 - 0.049202579500952j)
    np.testing.assert_almost_equal(Ey, 0.234284796808545 + 0.101857082148903j)
    np.testing.assert_almost_equal(Ez, 0)

    Ex, Ey, Ez = vwf.plane_vector_wave_function(x, y, z, kp, alpha, kz, 1)
    np.testing.assert_almost_equal(Ex, -0.140030336704405 + 0.322088344665751j)
    np.testing.assert_almost_equal(Ey, -0.067642363485058 + 0.155586406466850j)
    np.testing.assert_almost_equal(Ez, -0.442318214511318 - 0.192301179270512j)


def test_SVWF_against_prototype():
    Ex, Ey, Ez = vwf.spherical_vector_wave_function(x, y, z, omega, 1, tau, l, m)
    np.testing.assert_almost_equal(Ex, -0.010385224981764 + 0.018386955705419j)
    np.testing.assert_almost_equal(Ey, -0.005209637449869 + 0.011576972110819j)
    np.testing.assert_almost_equal(Ez, 0.002553743847975 + 0.001361996718920j)


def test_r_to_zero():
    Ex, Ey, Ez = vwf.spherical_vector_wave_function(0, 0, 0, omega, 1, 0, 1, -1)
    Ex2, Ey2, Ez2 = vwf.spherical_vector_wave_function(1e-3, 0, 0, omega, 1, 0, 1, -1)
    Ex3, Ey3, Ez3 = vwf.spherical_vector_wave_function(0, 1e-3, 0, omega, 1, 0, 1, -1)
    Ex4, Ey4, Ez4 = vwf.spherical_vector_wave_function(0, 0, 1e-3, omega, 1, 0, 1, -1)
    diff1 = (abs(Ex - Ex2) ** 2 + abs(Ey - Ey2) ** 2 + abs(Ez - Ez2) ** 2)
    diff2 = (abs(Ex - Ex3) ** 2 + abs(Ey - Ey3) ** 2 + abs(Ez - Ez3) ** 2)
    diff3 = (abs(Ex - Ex4) ** 2 + abs(Ey - Ey4) ** 2 + abs(Ez - Ez4) ** 2)
    assert diff1 < 1e-10
    assert diff2 < 1e-10
    assert diff3 < 1e-10

    Ex, Ey, Ez = vwf.spherical_vector_wave_function(0, 0, 0, omega, 1, 1, 1, -1)
    Ex2, Ey2, Ez2 = vwf.spherical_vector_wave_function(1e-3, 0, 0, omega, 1, 1, 1, -1)
    Ex3, Ey3, Ez3 = vwf.spherical_vector_wave_function(0, 1e-3, 0, omega, 1, 1, 1, -1)
    Ex4, Ey4, Ez4 = vwf.spherical_vector_wave_function(0, 0, 1e-3, omega, 1, 1, 1, -1)
    d1 = (abs(Ex - Ex2)**2 + abs(Ey - Ey2)**2 + abs(Ez - Ez2)**2) / (abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2)
    d2 = (abs(Ex - Ex3) ** 2 + abs(Ey - Ey3) ** 2 + abs(Ez - Ez3) ** 2) / (abs(Ex) ** 2 + abs(Ey) ** 2 + abs(Ez) ** 2)
    d3 = (abs(Ex - Ex4) ** 2 + abs(Ey - Ey4) ** 2 + abs(Ez - Ez4) ** 2) / (abs(Ex) ** 2 + abs(Ey) ** 2 + abs(Ez) ** 2)
    assert d1 < 1e-10
    assert d2 < 1e-10
    assert d3 < 1e-10

    Ex, Ey, Ez = vwf.spherical_vector_wave_function(0, 0, 0, omega, 1, 1, 1, 0)
    Ex2, Ey2, Ez2 = vwf.spherical_vector_wave_function(1e-3, 0, 0, omega, 1, 1, 1, 0)
    Ex3, Ey3, Ez3 = vwf.spherical_vector_wave_function(0, 1e-3, 0, omega, 1, 1, 1, 0)
    Ex4, Ey4, Ez4 = vwf.spherical_vector_wave_function(0, 0, 1e-3, omega, 1, 1, 1, 0)
    d1 = (abs(Ex - Ex2)**2 + abs(Ey - Ey2)**2 + abs(Ez - Ez2)**2) / (abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2)
    d2 = (abs(Ex - Ex3) ** 2 + abs(Ey - Ey3) ** 2 + abs(Ez - Ez3) ** 2) / (abs(Ex) ** 2 + abs(Ey) ** 2 + abs(Ez) ** 2)
    d3 = (abs(Ex - Ex4) ** 2 + abs(Ey - Ey4) ** 2 + abs(Ez - Ez4) ** 2) / (abs(Ex) ** 2 + abs(Ey) ** 2 + abs(Ez) ** 2)
    assert d1 < 1e-10
    assert d2 < 1e-10
    assert d3 < 1e-10


if __name__ == '__main__':
    test_PVWF_against_prototype()
    test_SVWF_against_prototype()
    test_r_to_zero()
