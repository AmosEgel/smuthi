# -*- coding: utf-8 -*-
"""Test the functions defined in vector_wave_functions.py."""

import unittest
import numpy as np
import smuthi.vector_wave_functions


class VWFtest(unittest.TestCase):
    def setUp(self):
        self.tau = 0
        self.l = 4
        self.m = -2
        self.pol = 0
        self.omega = 2 * 3.15 / 550
        self.kp = self.omega * 1.7
        self.alpha = 3.15/7
        self.kz = np.sqrt(self.omega**2 - self.kp**2 + 0j)
        self.x = 120
        self.y = -200
        self.z = 80
        self.z2 = 500
        self.lmax = 20

    def test_PVWF_against_prototype(self):
        Ex, Ey, Ez = smuthi.vector_wave_functions.plane_vector_wave_function(self.x, self.y, self.z, self.kp,
                                                                             self.alpha, self.kz, 0)
        self.assertAlmostEqual(Ex, -0.113172457895318 - 0.049202579500952j)
        self.assertAlmostEqual(Ey, 0.234284796808545 + 0.101857082148903j)
        self.assertAlmostEqual(Ez, 0)

        Ex, Ey, Ez = smuthi.vector_wave_functions.plane_vector_wave_function(self.x, self.y, self.z, self.kp,
                                                                             self.alpha, self.kz, 1)
        self.assertAlmostEqual(Ex, -0.140030336704405 + 0.322088344665751j)
        self.assertAlmostEqual(Ey, -0.067642363485058 + 0.155586406466850j)
        self.assertAlmostEqual(Ez, -0.442318214511318 - 0.192301179270512j)

    def test_SVWF_against_prototype(self):
        Ex, Ey, Ez = smuthi.vector_wave_functions.spherical_vector_wave_function(self.x, self.y, self.z, self.omega,
                                                                                 1, self.tau, self.l, self.m)
        self.assertAlmostEqual(Ex, -0.010385224981764 + 0.018386955705419j)
        self.assertAlmostEqual(Ey, -0.005209637449869 + 0.011576972110819j)
        self.assertAlmostEqual(Ez, 0.002553743847975 + 0.001361996718920j)

    def test_PVWF_in_SVWF(self):
        Ex, Ey, Ez = smuthi.vector_wave_functions.plane_vector_wave_function(self.x, self.y, self.z, self.kp,
                                                                             self.alpha, self.kz, self.pol)
        Ex2, Ey2, Ez2 = 0, 0, 0
        for tau in range(2):
            for l in range(1, self.lmax + 1):
                for m in range(-l, l + 1):
                    Nx, Ny, Nz = smuthi.vector_wave_functions.spherical_vector_wave_function(self.x, self.y, self.z,
                                                                                             self.omega, 1, tau, l, m)
                    B = smuthi.vector_wave_functions.transformation_coefficients_VWF(tau, l, m, self.pol, kp=self.kp,
                                                                                     kz=self.kz, dagger=True)
                    Ex2 += 4 * np.exp(-1j * m * self.alpha) * B * Nx
                    Ey2 += 4 * np.exp(-1j * m * self.alpha) * B * Ny
                    Ez2 += 4 * np.exp(-1j * m * self.alpha) * B * Nz

        self.assertAlmostEqual(Ex, Ex2)
        self.assertAlmostEqual(Ey, Ey2)
        self.assertAlmostEqual(Ez, Ez2)

    def test_SVWF_in_PVWF(self):
        Nx, Ny, Nz = smuthi.vector_wave_functions.spherical_vector_wave_function(self.x, self.y, self.z2,
                                                                                 self.omega, 3, self.tau, self.l,
                                                                                 self.m)
        kp_vec1 = np.linspace(0, 0.99, 1000) * self.omega
        kp_vec2 = 1j * np.linspace(0, -0.01, 1000) * self.omega + 0.99 * self.omega
        kp_vec3 = np.linspace(0, 0.02, 1000) * self.omega + (0.99 - 0.01j) * self.omega
        kp_vec4 = 1j * np.linspace(0, 0.01, 1000) * self.omega + (1.01 - 0.01j) * self.omega
        kp_vec5 = np.linspace(0, 6, 3000) * self.omega + 1.01 * self.omega

        kp_vec = np.concatenate([kp_vec1, kp_vec2, kp_vec3, kp_vec4, kp_vec5])
        kz_vec = np.sqrt(self.omega ** 2 - kp_vec ** 2 + 0j)
        alpha_vec = np.linspace(0, 2*np.pi, 500)
        kp_arr, alpha_arr = np.meshgrid(kp_vec, alpha_vec, indexing="ij")
        kz_arr = np.sqrt(self.omega**2 - kp_arr**2 + 0j)
        Nx2, Ny2, Nz2 = 0, 0, 0
        for pol in range(2):
            Ex, Ey, Ez = smuthi.vector_wave_functions.plane_vector_wave_function(self.x, self.y, self.z2, kp_arr,
                                                                                 alpha_arr, kz_arr, pol)
            B = smuthi.vector_wave_functions.transformation_coefficients_VWF(self.tau, self.l, self.m, pol, kp_vec,
                                                                             kz_vec)
            alpha_integrand = np.exp(1j * self.m * alpha_vec) * Ex
            kp_integrand = kp_vec / (self.omega * kz_vec) * B * np.trapz(alpha_integrand, x=alpha_vec, axis=1)
            Nx2 += 1 / (2 * np.pi) * np.trapz(kp_integrand, x=kp_vec)

            alpha_integrand = np.exp(1j * self.m * alpha_vec) * Ey
            kp_integrand = kp_vec / (self.omega * kz_vec) * B * np.trapz(alpha_integrand, x=alpha_vec, axis=1)
            Ny2 += 1 / (2 * np.pi) * np.trapz(kp_integrand, x=kp_vec)

            alpha_integrand = np.exp(1j * self.m * alpha_vec) * Ez
            kp_integrand = kp_vec / (self.omega * kz_vec) * B * np.trapz(alpha_integrand, x=alpha_vec, axis=1)
            Nz2 += 1 / (2 * np.pi) * np.trapz(kp_integrand, x=kp_vec)

        self.assertTrue(abs((Nx - Nx2) / Nx) < 1e-3)
        self.assertTrue(abs((Ny - Ny2) / Ny) < 1e-3)
        self.assertTrue(abs((Nz - Nz2) / Nz) < 1e-3)


if __name__ == '__main__':
    unittest.main()
