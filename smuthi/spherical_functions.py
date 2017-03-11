# -*- coding: utf-8 -*-

import numpy as np


def legendre_normalized(ct, st, lmax):
    """Return the normalized associated Legendre function P_l^m(cos \theta) and the angular functions
    \pi_l^m(cos \theta) and \tau_l^m(cos \theta), as defined in
    A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006.
    Two arguments (ct and st) are passed such that the function is valid for general complex arguments, while the branch
    cuts are defined by the user already in the definition of st.

    Input:
    ct      Array: cosine of theta (or kz/k)
    st      Array: sine of theta (or kp/k), need to have same dimension as ct, and st**2+ct**2=1 is assumed
    lmax    Integer: maximal multipole order

    Output:
    plm     List: plm[l][m] contains P_l^m(cos \theta). The entries of the list have same dimension as ct (and st)
    pilm    List: pilm[l][m] contains \pi_l^m(cos \theta).
    taulm   List: taulm[l][m] contains \tau_l^m(cos \theta).
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


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def double_factorial(n):
    if n in (0, 1):
        return 1
    else:
        return n * double_factorial(n - 2)
