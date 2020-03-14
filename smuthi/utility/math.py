"""This module contains several mathematical functions."""

# -*- coding: utf-8 -*-
import numpy as np
import scipy.special
import warnings
import smuthi.utility.memoizing as memo
import sys
import sympy
import math
from sympy.physics.quantum.spin import Rotation
try:
    from numba import cffi_support
    from pywigxjpf_ffi import ffi, lib
    import pywigxjpf_ffi
    cffi_support.register_module(pywigxjpf_ffi)
    nb_wig3jj = pywigxjpf_ffi.lib.wig3jj

    lib.wig_table_init(100,9)
    lib.wig_temp_init(100)
except:
    sys.stdout.write('No pywigxjpf installation found, '
    'using sympy implementaion of Wigner-3j symbols instead.\n'
    'In certain cases, this can significantly increase the simulation time.\n'
    'You can try "pip install pywigxjpf".\n')
    sys.stdout.flush()
    from sympy.physics.wigner import wigner_3j
    def nb_wig3jj(jj_1, jj_2, jj_3, mm_1, mm_2, mm_3):
        return float(wigner_3j(jj_1/2, jj_2/2, jj_3/2, mm_1/2, mm_2/2, mm_3/2))


def legendre_normalized(ct, st, lmax):
    r"""Return the normalized associated Legendre function :math:`P_l^m(\cos\theta)` and the angular functions
    :math:`\pi_l^m(\cos \theta)` and :math:`\tau_l^m(\cos \theta)`, as defined in
    `A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006
    <https://doi.org/10.1007/978-3-540-33697-6>`_.
    Two arguments (ct and st) are passed such that the function is valid for general complex arguments, while the branch
    cuts are defined by the user already in the definition of st.

    Args:
        ct (ndarray): cosine of theta (or kz/k)
        st (ndarray): sine of theta (or kp/k), need to have same dimension as ct, and st**2+ct**2=1 is assumed
        lmax (int): maximal multipole order

    Returns:
        - list plm[l][m] contains :math:`P_l^m(\cos \theta)`. The entries of the list have same dimension as ct (and st)
        - list pilm[l][m] contains :math:`\pi_l^m(\cos \theta)`.
        - list taulm[l][m] contains :math:`\tau_l^m(\cos \theta)`.
    """
    zr = ct - ct
    plm = [[zr for m in range(lmax + 1)] for l in range(lmax + 1)]
    pilm = [[zr for m in range(lmax + 1)] for l in range(lmax + 1)]
    taulm = [[zr for m in range(lmax + 1)] for l in range(lmax + 1)]
    pprimel0 = [zr for l in range(lmax + 1)]

    plm[0][0] += np.sqrt(2)/2
    plm[1][0] = np.sqrt(3/2) * ct
    pprimel0[1] = np.sqrt(3) * plm[0][0]
    taulm[0][0] = -st * pprimel0[0]
    taulm[1][0] = -st * pprimel0[1]

    for l in range(1, lmax):
        plm[l + 1][0] = (1 / (l + 1) * np.sqrt((2 * l + 1) * (2 * l + 3)) * ct * plm[l][0] -
                         l / (l + 1) * np.sqrt((2 * l + 3) / (2 * l - 1)) * plm[l-1][0])
        pprimel0[l + 1] = ((l + 1) * np.sqrt((2 * (l + 1) + 1) / (2 * (l + 1) - 1)) * plm[l][0] +
                           np.sqrt((2 * (l + 1) + 1) / (2 * (l + 1) - 1)) * ct * pprimel0[l])
        taulm[l + 1][0] = -st * pprimel0[l + 1]

    for m in range(1, lmax + 1):
        plm[m][m] = np.sqrt((2 * m + 1) / (2 * factorial(2 * m))) * double_factorial(2 * m - 1) * st**m
        pilm[m][m] = np.sqrt((2 * m + 1) / (2 * factorial(2 * m))) * double_factorial(2 * m - 1) * st**(m - 1)
        taulm[m][m] = m * ct * pilm[m][m]
        for l in range(m, lmax):
            plm[l + 1][m] = (np.sqrt((2 * l + 1) * (2 * l + 3) / ((l + 1 - m) * (l + 1 + m))) * ct * plm[l][m] -
                             np.sqrt((2 * l + 3) * (l - m) * (l + m) / ((2 * l - 1) * (l + 1 - m) * (l + 1 + m))) *
                             plm[l - 1][m])
            pilm[l + 1][m] = (np.sqrt((2 * l + 1) * (2 * l + 3) / (l + 1 - m) / (l + 1 + m)) * ct * pilm[l][m] -
                              np.sqrt((2 * l + 3) * (l - m) * (l + m) / (2 * l - 1) / (l + 1 - m) / (l + 1 + m)) *
                              pilm[l - 1][m])
            taulm[l + 1][m] = ((l + 1) * ct * pilm[l + 1][m] -
                               (l + 1 + m) * np.sqrt((2 * (l + 1) + 1) * (l + 1 - m) / (2 * (l + 1) - 1) / (l + 1 + m))
                               * pilm[l][m])

    return plm, pilm, taulm


if ('conda' in sys.version or 'continuum' in sys.version) and sys.platform.startswith('win32'):
    # Due to a bug in Anaconda build of scipy (see https://github.com/ContinuumIO/anaconda-issues/issues/1415 ),
    # the spherical Bessel and Hankel functions for complex arguments need to be defined using the following workaround:
    def spherical_bessel(n, x):
        """Spherical Bessel function. This is a wrapper for scipy.special.sph_jn to make it operate on numpy
        arrays.

        As soon as some bug for complex arguments is resolved, this can be replaced by scipy.special.spherical_jn.
        https://github.com/ContinuumIO/anaconda-issues/issues/1415

        Args:
            n (int): Order of spherical Bessel function
            x (array, complex or float): Argument for Bessel function

        Returns:
            Spherical Bessel function as array.
        """
        sphj = scipy.special.sph_jn
        if hasattr(x, "__iter__"):
            j_n = np.array([spherical_bessel(n, v) for v in x])
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
                j_n = sphj(n, x)[0][n]
        return j_n

    def spherical_hankel(n, x):
        """Spherical Hankel function of first kind.

        Args:
            n (int): Order of spherical Bessel function
            x (array, complex or float): Argument for Hankel function

        Returns:
            Spherical Hankel function as array.
        """
        sphj = scipy.special.sph_jn
        sphy = scipy.special.sph_yn
        if hasattr(x, "__iter__"):
            h_n = np.array([spherical_hankel(n, v) for v in x])
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
                h_n = sphj(n, x)[0][n] + 1j * sphy(n, x)[0][n]
        return h_n
else:
    spherical_bessel = scipy.special.spherical_jn
    def spherical_hankel(n, x):
        spherj = scipy.special.spherical_jn(n, x)
        sphery = scipy.special.spherical_yn(n, x)
        if hasattr(x, '__len__'):
            sphery[x==0] = np.nan
        return spherj + 1j * sphery


def dx_xj(n, x):
    r"""Derivative of :math:`x j_n(x)`, where :math:`j_n(x)` is the spherical Bessel function.

    Args:
        n (int): (n>0): Order of spherical Bessel function
        x (array, complex or float): Argument for spherical Bessel function

    Returns:
        Derivative :math:`\partial_x(x j_n(x))` as array.
    """
    res = x * spherical_bessel(n - 1, x) - n * spherical_bessel(n, x)
    return res


def dx_xh(n, x):
    r"""Derivative of :math:`x h_n(x)`, where :math:`h_n(x)` is the spherical Hankel function.

    Args:
        n (int): (n>0): Order of spherical Bessel function
        x (array, complex or float): Argument for spherical Hankel function

    Returns:
        Derivative :math:`\partial_x(x h_n(x))` as array.
    """
    res = x * spherical_hankel(n - 1, x) - n * spherical_hankel(n, x)
    return res


@memo.Memoize
def factorial(n):
    """Return factorial.

    Args:
        n (int): Argument (non-negative)

    Returns:
        Factorial of n
    """
    assert type(n) == int and n >= 0
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


@memo.Memoize
def double_factorial(n):
    """Return double factorial.

    Args:
        n (int): Argument (non-negative)

    Returns:
        Double factorial of n
    """
    assert type(n) == int and n >= 0
    if n in (0, 1):
        return 1
    else:
        return n * double_factorial(n - 2)


def wigner_d(l, m, m_prime, beta, wdsympy=False):
    """Computation of Wigner-d-functions for the rotation of a T-matrix
    
    Args:
        l (int):          Degree :math:`l` (1, ..., lmax)
        m (int):          Order :math:`m` (-min(l,mmax),...,min(l,mmax))
        m_prime (int):    Order :math:`m_prime` (-min(l,mmax),...,min(l,mmax))
        beta (float):     Second Euler angle in rad
        wdsympy (bool):   If True, Wigner-d-functions come from the sympy toolbox 
        
    Returns:
        real value of Wigner-d-function
    """
    wig_d = np.zeros(l + 1, dtype=complex)
    
    if wdsympy == False:
    
        if beta < 0:
            aa = m
            bb = m_prime
            m = bb
            m_prime = aa
           
        if m == 0 and m_prime == 0:
            for nn in range(1, l + 1):
                wig_d[nn] = sympy.legendre_poly(nn, np.cos(beta))          
        else:
            # recursion formulation (Mishchenko, Scattering, Absorption and Emission of Light by small Particles, p.365 (B.22 - B.24))
            l_min = max(abs(m), abs(m_prime))
            wig_d[l_min - 1] = 0
            if m_prime >= m:
                zeta = 1
            else:
                zeta = (-1) ** (m - m_prime)
        
            wig_d[l_min] = (zeta * 2.0 ** (-l_min) * (factorial(2 * l_min) / (factorial(abs(m - m_prime)) 
                                                                              * factorial(abs(m + m_prime)))) ** 0.5
                            * (1 - np.cos(beta)) ** (abs(m - m_prime) / 2) 
                            * (1 + np.cos(beta)) ** (abs(m + m_prime) / 2 ))
    
            for ll in range(l_min, l):
                wig_d[ll + 1] = (((2 * ll + 1) * (ll * (ll + 1) * np.cos(beta) - m * m_prime) * wig_d[ll] 
                                - (ll + 1) * (ll ** 2 - m ** 2) ** 0.5 * (ll ** 2 - m_prime ** 2) ** 0.5 
                                * wig_d[ll - 1]) / (ll * ((ll + 1) ** 2 - m ** 2) ** 0.5 
                                                    * ((ll + 1) ** 2 - m_prime ** 2) ** 0.5))
    
    else:
        wig_d[l] = complex(Rotation.d(l, m, m_prime, beta).doit())
      
    return wig_d[l].real


def wigner_D(l , m, m_prime, alpha, beta, gamma, wdsympy=False):
    """Computation of Wigner-D-functions for the rotation of a T-matrix
         
    Args:
        l (int):          Degree :math:`l` (1, ..., lmax)
        m (int):          Order :math:`m` (-min(l,mmax),...,min(l,mmax))
        m_prime (int):    Order :math:`m_prime` (-min(l,mmax),...,min(l,mmax))
        alpha (float):    First Euler angle in rad
        beta (float):     Second Euler angle in rad
        gamma (float):    Third Euler angle in rad
        wdsympy (bool):   If True, Wigner-d-functions come from the sympy toolbox
        
    Returns:
        single complex value of Wigner-D-function 
    """       
    # Doicu, Light Scattering by Systems of Particles, p. 271ff (B.33ff)   
    if m >= 0 and m_prime >= 0:
        delta_m_mprime = 1
    elif m >= 0 and m_prime < 0:
        delta_m_mprime = (-1) ** m_prime
    elif m < 0 and m_prime >= 0:
        delta_m_mprime = (-1) ** m
    elif m < 0 and m_prime < 0:
        delta_m_mprime = (-1) ** (m + m_prime)
        
    wig_D = ((-1) ** (m + m_prime) * np.exp(1j * m * alpha) * delta_m_mprime * wigner_d(l, m, m_prime, beta, wdsympy) 
            * np.exp(1j * m_prime * gamma))

    # Mishchenko, Scattering, Absorption and Emission of Light by small Particles, p.367 (B.38)
#    wig_D = np.exp(-1j * m * alpha) * wigner_d(l, m, m_prime, beta) * np.exp(-1j * m_prime * gamma)    
   
    return wig_D


def rotation_matrix(alpha=None, beta=None, gamma=None, euler_angles=None):
    if euler_angles is not None:
        alpha = euler_angles[0]
        beta = euler_angles[1]
        gamma = euler_angles[2]
    rotation_matrix_3 = [[np.cos(gamma), np.sin(gamma), 0], [- np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]
    rotation_matrix_2 = [[np.cos(beta), 0, - np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]]
    rotation_matrix_1 = [[np.cos(alpha), np.sin(alpha), 0], [- np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]] 
    return np.dot(rotation_matrix_3, np.dot(rotation_matrix_2, rotation_matrix_1))


def vector_rotation(r, alpha=None, beta=None, gamma=None, euler_angles=None):
    return np.dot(rotation_matrix(alpha, beta, gamma, euler_angles), r)


def inverse_vector_rotation(r, alpha=None, beta=None, gamma=None, euler_angles=None):
    return np.dot(np.linalg.inv(rotation_matrix(alpha, beta, gamma, euler_angles)), r)
