# -*- coding: utf-8 -*-
"""This module contains functionality that has to do with the spherical 
coordinates (k, alpha, beta) or cylindrical coordinates 
(k_parallel, alpha, k_z) of a wave-vector or a far field direction.
It also contains the definition of complex contours in k_parallel for 
Sommerfeld integration."""

import numpy as np
from smuthi.fields import angular_frequency

"""The default arrays for k_parrallel, azimuthal_angles and polar_angles are 
used in Sommerfeld integrals or in plane wave expansions whenever no other
arrays for the specification of the wavevectors are explicitly stated."""
default_k_parallel = None
default_azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180
default_polar_angles = np.arange(0, 181, 1, dtype=float) * np.pi / 180


def complex_contour(vacuum_wavelength, neff_waypoints, neff_resolution):
    """Construct an array of complex in-plane wavenumbers (i.e., the radial
    component of the cylindrical coordinates of the wave-vector). 
    This is used for the plane wave expansion of fields and for Sommerfeld 
    integrals. Complex contours are used to improve numerical stability 
    (see section 3.10.2.1 of [Egel 2018 dissertation]).
    
    Args:
        vacuum_wavelength (float):        Vacuum wavelength :math:`\lambda` (length)
        neff_waypoints (list or ndarray): Corner points through which the contour runs
                                          This quantity is dimensionless (effective
                                          refractive index, will be multiplied by vacuum
                                          wavenumber) 
        neff_resolution(float):           Resolution of contour, again in terms of
                                          effective refractive index
 
    Returns:
        Array :math:`\kappa_i` of in-plane wavenumbers (inverse length)
    
    """
    neff_segments = []
    for i in range(len(neff_waypoints)-1):
        abs_dneff = abs(neff_waypoints[i + 1] - neff_waypoints[i])
        neff_segments.append(neff_waypoints[i] + np.arange(0, 1 + neff_resolution/abs_dneff/2, neff_resolution/abs_dneff, 
                                                           dtype=complex) 
                             * (neff_waypoints[i + 1] - neff_waypoints[i]))
    return np.concatenate(neff_segments) * angular_frequency(vacuum_wavelength)


def set_default_k_parallel(vacuum_wavelength, neff_waypoints=None, 
                           neff_resolution=1e-2, neff_max=None, neff_imag=0.01):
    if neff_waypoints is None:
        neff_waypoints = (0, 0.8, 0.8-1j*neff_imag, neff_max-1j*neff_imag, neff_max)
    global default_k_parallel
    kpar = complex_contour(vacuum_wavelength, neff_waypoints, neff_resolution)
    default_k_parallel = kpar
 
 
def k_z(k_parallel=None, n_effective=None, k=None, omega=None, vacuum_wavelength=None, refractive_index=None):
    """z-component :math:`k_z=\sqrt{k^2-\kappa^2}` of the wavevector. The branch cut is defined such that the imaginary
    part is not negative, compare section 2.3.1 of [Egel 2018 dissertation]. 
    Not all of the arguments need to be specified.
 
    Args:
        k_parallel (numpy ndarray):     In-plane wavenumber :math:`\kappa` (inverse length)
        n_effective (numpy ndarray):    Effective refractive index :math:`n_\mathrm{eff}`
        k (float):                      Wavenumber (inverse length)
        omega (float):                  Angular frequency :math:`\omega` or vacuum wavenumber (inverse length, c=1)
        vacuum_wavelength (float):      Vacuum wavelength :math:`\lambda` (length)
        refractive_index (complex):     Refractive index :math:`n_i` of material
 
    Returns:
        z-component :math:`k_z` of wavenumber with non-negative imaginary part (inverse length)
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
