# -*- coding: utf-8 -*-
"""Test the orthogonality relations of the SVWFs."""

import unittest

import smuthi.fields.vector_wave_functions as vwf
import smuthi.utility.math as sf
import smuthi.fields.expansions as fldex
import numpy as np
import scipy.integrate

lmax = 3


def SVWF_solid_angle_integral(tau1, l1, m1, nu1, tau2, l2, m2, nu2, r, n, wavelength):
    k = 2*np.pi / wavelength * n

    def theta_function(theta):

        def phi_function(phi):
            x = np.sin(theta) * np.cos(phi) * r
            y = np.sin(theta) * np.sin(phi) * r
            z = np.cos(theta) * r
            E1x, E1y, E1z = vwf.spherical_vector_wave_function(x, y, z, k, nu1, tau1, l1, m1)
            E2x, E2y, E2z = vwf.spherical_vector_wave_function(x, y, z, k, nu2, tau2, l2, m2)
            return E1x * E2x + E1y * E2y + E1z * E2z

        val, err = scipy.integrate.quadrature(phi_function, 0, 2*np.pi, vec_func=False)
        return np.sin(theta) * val

    val, err = scipy.integrate.quadrature(theta_function, 0, np.pi, vec_func=False)
    return val


def test_different_tau_zero():
    tau1 = 0; tau2 = 1; nu1 = 1; nu2 = 1;
    r = 100; n = 1.5; wl = 550
    for l1 in range(1, lmax+1):
        for m1 in range(-l1, l1+1):
            for l2 in range(1, lmax+1):
                for m2 in range(-l2, l2+1):
                    surface_integral = SVWF_solid_angle_integral(tau1, l1, m1, nu1, tau2, l2, m2, nu2, r, n, wl)
                    print("tau1=%i, tau2=%i, l1=%i, l2=%i, m1=%i, m2=%i, nu1=%i, nu2=%i"%(tau1, tau2, l1, l2, m1, m2, nu1, nu2))
                    print("Orthogonality relation: %e, Expected: 0"%surface_integral)
                    assert (surface_integral < 1e-8)
                    print("OK.\n")


def test_tau1_bessel():
    tau1 = 0; tau2 = 0; nu1 = 1; nu2 = 1;
    r = 100; n = 1.5; wl = 550
    k = 2 * np.pi / wl * n
    z = [0, sf.spherical_bessel, 0, sf.spherical_hankel]

    # non-zero case
    for l1 in range(1, lmax+1):
        for m1 in range(0, l1+1):
            l2 = l1
            m2 = -m1
            surface_integral = SVWF_solid_angle_integral(tau1, l1, m1, nu1, tau2, l2, m2, nu2, r, n, wl)
            if m1 == -m2 and l1 == l2:
                expected = np.pi * z[nu1](l1, k * r) * z[nu2](l1, k * r)
            else:
                expected = 0
            print("tau1=%i, tau2=%i, l1=%i, l2=%i, m1=%i, m2=%i, nu1=%i, nu2=%i"%(tau1, tau2, l1, l2, m1, m2, nu1, nu2))
            print("Surface integral: %e+%ej, Expected: %e"%(surface_integral.real, surface_integral.imag, expected))
            np.testing.assert_almost_equal(surface_integral, expected)
            print("OK.\n")

    # zero case
    for l1 in range(1, lmax+1):
        for m1 in range(-l1, l1+1):
            for l2 in range(1, lmax+1):
                for m2 in range(-l2, l2+1):
                    if m1 == -m2 and l1 == l2:
                        continue
                    else:
                        expected = 0
                        surface_integral = SVWF_solid_angle_integral(tau1, l1, m1, nu1, tau2, l2, m2, nu2, r, n, wl)
                        print("tau1=%i, tau2=%i, l1=%i, l2=%i, m1=%i, m2=%i, nu1=%i, nu2=%i"%(tau1, tau2, l1, l2, m1, m2, nu1, nu2))
                        print("Surface integral: %e, Expected: %e"%(np.abs(surface_integral), expected))
                        np.testing.assert_almost_equal(surface_integral, expected)
                        print("OK.\n")


def test_tau2_expression():
    tau1 = 1; tau2 = 1; nu1 = 1; nu2 = 1;
    r = 100; n = 1.5; wl = 550
    k = 2 * np.pi / wl * n
    z = [0, sf.spherical_bessel, 0, sf.spherical_hankel]
    dxxz = [0, sf.dx_xj, 0, sf.dx_xh]

    # non-zero case
    for l in range(1, lmax+1):
        for m in range(0, l+1):
            l1 = l
            l2 = l
            m1 = m
            m2 = -m
            surface_integral = SVWF_solid_angle_integral(tau1, l1, m1, nu1, tau2, l2, m2, nu2, r, n, wl)
            expected = (np.pi / (k*r)**2 *
                        (l * (l + 1) * z[nu1](l, k*r) * z[nu2](l, k*r) + dxxz[nu1](l, k*r) * dxxz[nu2](l, k*r)))
            print(
                "tau1=%i, tau2=%i, l1=%i, l2=%i, m1=%i, m2=%i, nu1=%i, nu2=%i" % (tau1, tau2, l1, l2, m1, m2, nu1, nu2))
            np.testing.assert_almost_equal(surface_integral, expected)
            print("Surface integral: %e+%ej, Expected: %e"%(surface_integral.real, surface_integral.imag, expected))
            print("OK.\n")

    # zero case
    for l1 in range(1, lmax+1):
        for m1 in range(-l1, l1+1):
            for l2 in range(1, lmax+1):
                for m2 in range(-l2, l2+1):
                    if m1 == -m2 and l1 == l2:
                        continue
                    else:
                        expected = 0
                        surface_integral = SVWF_solid_angle_integral(tau1, l1, m1, nu1, tau2, l2, m2, nu2, r, n, wl)
                        print("tau1=%i, tau2=%i, l1=%i, l2=%i, m1=%i, m2=%i, nu1=%i, nu2=%i"
                              %(tau1, tau2, l1, l2, m1, m2, nu1, nu2))
                        np.testing.assert_almost_equal(surface_integral, expected)
                        print("Surface integral: %e, Expected: %e"%(np.abs(surface_integral), expected))
                        print("OK.\n")


if __name__ == '__main__':
    test_tau2_expression()
    test_tau1_bessel()
    test_different_tau_zero()
