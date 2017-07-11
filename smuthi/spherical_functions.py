# -*- coding: utf-8 -*-

import numpy as np
import scipy.special
import warnings


def legendre_normalized(ct, st, lmax):
    r"""Return the normalized associated Legendre function :math:`P_l^m(\cos\theta)` and the angular functions
    :math:`\pi_l^m(\cos \theta)` and :math:`\tau_l^m(\cos \theta)`, as defined in
    `A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006
    <https://doi.org/10.1007/978-3-540-33697-6>`_.
    Two arguments (ct and st) are passed such that the function is valid for general complex arguments, while the branch
    cuts are defined by the user already in the definition of st.

    Args:
        ct (array): cosine of theta (or kz/k)
        st (array): sine of theta (or kp/k), need to have same dimension as ct, and st**2+ct**2=1 is assumed
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
