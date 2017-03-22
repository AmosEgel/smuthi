# -*- coding: utf-8 -*-
import numpy as np


class ComplexContour:
    """Trajectory of n_effective in the complex plane for the evaluation of Sommerfeld integrals."""
    def __init__(self, neff_waypoints=[0, 1], neff_discretization=1e-2):
        # n_effective waypoints, that is, points through which the contour goes (linear between them)
        # k_parallel = n_effective * omega
        self.neff_waypoints = neff_waypoints

        # Discretization between the waypoints. Either as an array or as a scalar (uniform for all segments)
        self.neff_discretization = neff_discretization

    def neff(self):
        """Return numpy-array of n_effective values that define the contour."""
        neff_segments = []
        for i in range(len(self.neff_waypoints)-1):
            if hasattr(self.neff_discretization, "__len__"):
                npoints = abs(self.neff_waypoints[i+1] - self.neff_waypoints[i]) / self.neff_discretization[i]
            else:
                npoints = abs(self.neff_waypoints[i + 1] - self.neff_waypoints[i]) / self.neff_discretization
            neff_segments.append(self.neff_waypoints[i] + np.linspace(0, 1, num=npoints, endpoint=True, dtype=complex) *
                                 (self.neff_waypoints[i + 1] - self.neff_waypoints[i]))
        return np.concatenate(neff_segments)


def k_z(k_parallel=None, n_effective=None, k=None, omega=None, vacuum_wavelength=None, refractive_index=None):
    """Return z-component of wavevector with k_z.imag >= 0

    Input:
    k_parallel:         in-plane wavenumber (inverse length)
    n_effective:        k_parallel = n_effective * omega
    k:                  wavenumber (inverse length)
    omega:              angular frequency (inverse length, c=1)
    vacuum_wavelength:  (length)
    refractive_index:   refractive index of material in which k_z is evaluated
    """
    if k_parallel is None:
        if omega is None:
            omega = angular_frequency(vacuum_wavelength)
        k_parallel = n_effective * omega

    if k is None:
        if omega is None:
            omega = angular_frequency(vacuum_wavelength)
        k = refractive_index * omega

    kz = np.sqrt(k ** 2 - k_parallel ** 2 + 0j)
    kz = (kz.imag >= 0) * kz + (kz.imag < 0) * (-kz)  # Branch cut such to prohibit negative imaginary
    return kz


def angular_frequency(vacuum_wavelength):
    """Angular frequency in the units of c=1 (time=length)"""
    return 2 * np.pi / vacuum_wavelength