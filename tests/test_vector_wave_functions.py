# -*- coding: utf-8 -*-
"""Test the functions defined in vector_wave_functions.py."""

import numpy as np
import smuthi.vector_wave_functions


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
    Ex, Ey, Ez = smuthi.vector_wave_functions.plane_vector_wave_function(x, y, z, kp, alpha, kz, 0)
    
    np.testing.assert_almost_equal(Ex, -0.113172457895318 - 0.049202579500952j)
    np.testing.assert_almost_equal(Ey, 0.234284796808545 + 0.101857082148903j)
    np.testing.assert_almost_equal(Ez, 0)

    Ex, Ey, Ez = smuthi.vector_wave_functions.plane_vector_wave_function(x, y, z, kp, alpha, kz, 1)
    np.testing.assert_almost_equal(Ex, -0.140030336704405 + 0.322088344665751j)
    np.testing.assert_almost_equal(Ey, -0.067642363485058 + 0.155586406466850j)
    np.testing.assert_almost_equal(Ez, -0.442318214511318 - 0.192301179270512j)

def test_SVWF_against_prototype():
    Ex, Ey, Ez = smuthi.vector_wave_functions.spherical_vector_wave_function(x, y, z, omega, 1, tau, l, m)
    np.testing.assert_almost_equal(Ex, -0.010385224981764 + 0.018386955705419j)
    np.testing.assert_almost_equal(Ey, -0.005209637449869 + 0.011576972110819j)
    np.testing.assert_almost_equal(Ez, 0.002553743847975 + 0.001361996718920j)

def test_PVWF_in_SVWF():
    Ex, Ey, Ez = smuthi.vector_wave_functions.plane_vector_wave_function(x, y, z, kp, alpha, kz, pol)
    Ex2, Ey2, Ez2 = 0, 0, 0
    for tau in range(2):
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                Nx, Ny, Nz = smuthi.vector_wave_functions.spherical_vector_wave_function(x, y, z, omega, 1, tau, l, m)
                B = smuthi.vector_wave_functions.transformation_coefficients_VWF(tau, l, m, pol, kp=kp, kz=kz,
                                                                                 dagger=True)
                Ex2 += 4 * np.exp(-1j * m * alpha) * B * Nx
                Ey2 += 4 * np.exp(-1j * m * alpha) * B * Ny
                Ez2 += 4 * np.exp(-1j * m * alpha) * B * Nz

    np.testing.assert_almost_equal(Ex, Ex2)
    np.testing.assert_almost_equal(Ey, Ey2)
    np.testing.assert_almost_equal(Ez, Ez2)

def test_SVWF_in_PVWF():
    Nx, Ny, Nz = smuthi.vector_wave_functions.spherical_vector_wave_function(x, y, z2, omega, 3, tau, l, m)
    kp_vec1 = np.linspace(0, 0.99, 1000) * omega
    kp_vec2 = 1j * np.linspace(0, -0.01, 1000) * omega + 0.99 * omega
    kp_vec3 = np.linspace(0, 0.02, 1000) * omega + (0.99 - 0.01j) * omega
    kp_vec4 = 1j * np.linspace(0, 0.01, 1000) * omega + (1.01 - 0.01j) * omega
    kp_vec5 = np.linspace(0, 6, 3000) * omega + 1.01 * omega

    kp_vec = np.concatenate([kp_vec1, kp_vec2, kp_vec3, kp_vec4, kp_vec5])
    kz_vec = np.sqrt(omega ** 2 - kp_vec ** 2 + 0j)
    alpha_vec = np.linspace(0, 2*np.pi, 500)
    kp_arr, alpha_arr = np.meshgrid(kp_vec, alpha_vec, indexing="ij")
    kz_arr = np.sqrt(omega**2 - kp_arr**2 + 0j)
    Nx2, Ny2, Nz2 = 0, 0, 0

    for pol in range(2):
        Ex, Ey, Ez = smuthi.vector_wave_functions.plane_vector_wave_function(x, y, z2, kp_arr, alpha_arr, kz_arr, pol)
        B = smuthi.vector_wave_functions.transformation_coefficients_VWF(tau, l, m, pol, kp_vec, kz_vec)
        alpha_integrand = np.exp(1j * m * alpha_vec) * Ex
        kp_integrand = kp_vec / (omega * kz_vec) * B * np.trapz(alpha_integrand, x=alpha_vec, axis=1)
        Nx2 += 1 / (2 * np.pi) * np.trapz(kp_integrand, x=kp_vec)

        alpha_integrand = np.exp(1j * m * alpha_vec) * Ey
        kp_integrand = kp_vec / (omega * kz_vec) * B * np.trapz(alpha_integrand, x=alpha_vec, axis=1)
        Ny2 += 1 / (2 * np.pi) * np.trapz(kp_integrand, x=kp_vec)

        alpha_integrand = np.exp(1j * m * alpha_vec) * Ez
        kp_integrand = kp_vec / (omega * kz_vec) * B * np.trapz(alpha_integrand, x=alpha_vec, axis=1)
        Nz2 += 1 / (2 * np.pi) * np.trapz(kp_integrand, x=kp_vec)

    assert abs((Nx - Nx2) / Nx) < 1e-3
    assert abs((Ny - Ny2) / Ny) < 1e-3
    assert abs((Nz - Nz2) / Nz) < 1e-3


if __name__ == '__main__':
    test_PVWF_against_prototype()
    test_PVWF_in_SVWF()
    test_SVWF_against_prototype()
    test_SVWF_in_PVWF()
