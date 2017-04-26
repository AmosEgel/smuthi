# -*- coding: utf-8 -*-
"""Test the functions defined in coordinates.py."""

import unittest
import smuthi.coordinates
import matplotlib.pyplot as plt


def test_Contour_against_prototype():
    neff_wpts = [0, 1.49, 1.5 - 0.05j, 1.8 - 0.05j, 1.81, 4]
    neff_discr = [1e-2, 1e-2, 1e-3, 1e-3, 1e-2]
    contour_object = smuthi.coordinates.ComplexContour(neff_waypoints=neff_wpts, neff_discretization=neff_discr)
    neff_discr = 1e-2
    contour_object2 = smuthi.coordinates.ComplexContour(neff_waypoints=neff_wpts, neff_discretization=neff_discr)


def test_kz():
    neff_wpts = [0, 1.49, 1.5 - 0.05j, 1.8 - 0.05j, 1.81, 4]
    neff_discr = [1e-2, 1e-2, 1e-3, 1e-3, 1e-2]
    contour_object = smuthi.coordinates.ComplexContour(neff_waypoints=neff_wpts, neff_discretization=neff_discr)
    wl = 550
    n = 1.62 + 0.01j
    omega = smuthi.coordinates.angular_frequency(wl)
    k = omega * n
    kp = contour_object.neff() * omega
    kz1 = smuthi.coordinates.k_z(k_parallel=kp, k=k)
    kz2 = smuthi.coordinates.k_z(n_effective=contour_object.neff(), vacuum_wavelength=wl, refractive_index=n)
    assert all(kz1 == kz2)
    assert all(kz1 ** 2 + kp ** 2 - k ** 2 < abs(k ** 2 * 1e-14))


if __name__ == '__main__':
    test_Contour_against_prototype()
    test_kz()
