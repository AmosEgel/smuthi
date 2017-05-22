# -*- coding: utf-8 -*-

import numpy as np
import sympy.physics.wigner
import sympy
import smuthi.spherical_functions as sf


def plane_vector_wave_function(x, y, z, kp, alpha, kz, pol):
    """Return the electric field components of plane wave (PVWF).

    The input arrays should have one of the following dimensions
    x,y,z: (N x 1) matrix
    kp,alpha,kz: (1 x M) matrix
    Ex, Ey, Ez: (M x N) matrix

    or
    x,y,z: (M x N) matrix
    kp,alpha,kz: scalar
    Ex, Ey, Ez: (M x N) matrix

    Input:
    x       numpy-array: x-coordinate of position where to test the field (length unit)
    y       numpy-array: y-coordinate of position where to test the field
    z       numpy-array: z-coordinate of position where to test the field
    kp      numpy-array: parallel component of k-vector (inverse length unit)
    alpha   numpy-array: azimthal angle of k-vector (rad)
    kz      numpy-array: z-component of k-vector (inverse length unit)
    pol     integer: (0=TE, 1=TM)

    Output:
    Ex      numpy-array: x-coordinate of PVWF electric field
    Ey      numpy-array: y-coordinate of PVWF electric field
    Ez      numpy-array: z-coordinate of PVWF electric field
    """
    k = np.sqrt(kp**2 + kz**2)
    kx = kp * np.cos(alpha)
    ky = kp * np.sin(alpha)

    scalar_wave = np.exp(1j * (kx * x + ky * y + kz * z))

    if pol == 0:
        Ex = -np.sin(alpha) * scalar_wave
        Ey = np.cos(alpha) * scalar_wave
        Ez = scalar_wave - scalar_wave
    elif pol == 1:
        Ex = np.cos(alpha) * kz / k * scalar_wave
        Ey = np.sin(alpha) * kz / k * scalar_wave
        Ez = -kp / k * scalar_wave
    else:
        raise ValueError('Polarization must be 0 (TE) or 1 (TM)')

    return Ex, Ey, Ez


def spherical_vector_wave_function(x, y, z, k, nu, tau, l, m):
    """Return the electric field components of spherical vector wave function (SVWF). The conventions are chosen
    according to
    A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006.

    Input:
    x       numpy-array: x-coordinate of position where to test the field (length unit)
    y       numpy-array: y-coordinate of position where to test the field
    z       numpy-array: z-coordinate of position where to test the field
    k       scalar: wavenumber (inverse length unit)
    nu      integer: 1 for regular waves, 3 for outgoing waves
    tau     integer: spherical polarization, 0 for spherical TE and 1 for spherical TM
    l       integer: l=1,... multipole degree (polar quantum number)
    m       integer: m=-l,...,l multipole order (azimuthal quantum number)

    Output:
    Ex      numpy-array: x-coordinate of SVWF electric field
    Ey      numpy-array: y-coordinate of SVWF electric field
    Ez      numpy-array: z-coordinate of SVWF electric field
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    # unit vector in r-direction
    er_x = x / r
    er_y = y / r
    er_z = z / r

    # unit vector in theta-direction
    eth_x = np.cos(theta) * np.cos(phi)
    eth_y = np.cos(theta) * np.sin(phi)
    eth_z = -np.sin(theta)

    # unit vector in phi-direction
    eph_x = -np.sin(phi)
    eph_y = np.cos(phi)
    eph_z = x - x

    cos_thet = np.cos(theta)
    sin_thet = np.sin(theta)
    plm_list, pilm_list, taulm_list = sf.legendre_normalized(cos_thet, sin_thet, l)
    plm = plm_list[l][abs(m)]
    pilm = pilm_list[l][abs(m)]
    taulm = taulm_list[l][abs(m)]

    kr = k * r
    if nu == 1:
        bes = sf.spherical_bessel(l, kr)
        dxxz = sf.dx_xj(l, kr)
    elif nu == 3:
        bes = sf.spherical_hankel(l, kr)
        dxxz = sf.dx_xh(l, kr)
    else:
        raise ValueError('nu must be 1 (regular SVWF) or 3 (outgoing SVWF)')

    eimphi = np.exp(1j * m * phi)
    prefac = 1/np.sqrt(2 * l * (l + 1))
    if tau == 0:
        Ex = prefac * bes * (1j * m * pilm * eth_x - taulm * eph_x) * eimphi
        Ey = prefac * bes * (1j * m * pilm * eth_y - taulm * eph_y) * eimphi
        Ez = prefac * bes * (1j * m * pilm * eth_z - taulm * eph_z) * eimphi
    elif tau == 1:
        Ex = prefac * (l * (l + 1) * bes / kr * plm * er_x +
                       dxxz / kr * (taulm * eth_x + 1j * m * pilm * eph_x)) * eimphi
        Ey = prefac * (l * (l + 1) * bes / kr * plm * er_y +
                       dxxz / kr * (taulm * eth_y + 1j * m * pilm * eph_y)) * eimphi
        Ez = prefac * (l * (l + 1) * bes / kr * plm * er_z +
                       dxxz / kr * (taulm * eth_z + 1j * m * pilm * eph_z)) * eimphi
    else:
        raise ValueError('tau must be 0 (spherical TE) or 1 (spherical TM)')

    return Ex, Ey, Ez


def transformation_coefficients_VWF(tau, l, m, pol, kp=None, kz=None, pilm_list=None, taulm_list=None, dagger=False):
    """Return the transformation coefficients B to expand SVWF in PVWF and vice versa. See theory part of documentation.

    Input:
    tau         integer: SVWF polarization, 0 for spherical TE, 1 for spherical TM
    l           integaer: l=1,... SVWF multipole degree
    m           integaer: m=-l,...,l SVWF multipole order
    pol         integer: PVWF polarization, 0 for TE, 1 for TM
    kp          complex numpy-array: PVWF in-plane wavenumbers
    kz          complex numpy-array: PVWF out-of-plane wavenumbers
    pilm_list   2D list numpy-arrays: alternatively to kp and kz, pilm and taulm as generated with legendre_normalized
                can directly be handed
    taulm_list  2D list numpy-arrays: alternatively to kp and kz, pilm and taulm as generated with legendre_normalized
                can directly be handed
    dagger      logical, switch on when expanding PVWF in SVWF and off when expanding SVWF in PVWF
    """
    if pilm_list is None:
        k = np.sqrt(kp**2 + kz**2)
        ct = kz / k
        st = kp / k
        plm_list, pilm_list, taulm_list = sf.legendre_normalized(ct, st, l)

    if tau == pol:
        sphfun = taulm_list[l][abs(m)]
    else:
        sphfun = m * pilm_list[l][abs(m)]

    if dagger:
        if pol == 0:
            prefac = -1 / (-1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * (-1j)
        elif pol == 1:
            prefac = -1 / (-1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * 1
        else:
            raise ValueError('pol must be 0 (TE) or 1 (TM)')
    else:
        if pol == 0:
            prefac = -1 / (1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * (1j)
        elif pol ==1:
            prefac = -1 / (1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * 1
        else:
            raise ValueError('pol must be 0 (TE) or 1 (TM)')

    B = prefac * sphfun

    return B


def translation_coefficients_svwf(l1, m1, l2, m2, k, d, sph_hankel=None, legendre=None, exp_immphi=None):
    """Return the coefficients of the translation operator for the expansion of an outgoing spherical wave in terms of
    regular spherical waves with respect to a different origin.
    The output is a tuple (A,B), where the translation operator
        trans = \delta_pp' * A + (1-\delta_pp') * B

    Input:
    l1          integaer: l=1,...: Original wave's SVWF multipole degree
    m1          integaer: m=-l,...,l: Original wave's SVWF multipole order
    l2          integaer: l=1,...: Partial wave's SVWF multipole degree
    m2          integaer: m=-l,...,l: Partial wave's SVWF multipole order
    k           complex: wavenumber (inverse length unit)
    d           translation vectors in format [dx, dy, dz] (length unit)
                dx, dy, dz can be scalars or ndarrays
    sph_hankel  list: sph_hankel[i] contains the spherical hankel funciton of degree i, evaluated at k*d where d is the
                norm of the distance vector(s)
    legendre    list of lists: legendre[l][m] contains the legendre function of order l and degree m, evaluated at
                cos(theta) where theta is the polar angle(s) of the distance vector(s)
    """
    # spherical coordinates of d:
    dd = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

    if exp_immphi is None:
        phid = np.arctan2(d[1], d[0])
        eimph = np.exp(1j * (m1 - m2) * phid)
    else:
        eimph = exp_immphi[m1][m2]

    if sph_hankel is None:
        sph_hankel = [sf.spherical_hankel(n, k * dd) for n in range(l1 + l2 + 1)]

    if legendre is None:
        costthetd = d[2] / dd
        sinthetd = np.sqrt(d[0] ** 2 + d[1] ** 2) / dd
        legendre, _, _ = sf.legendre_normalized(costthetd, sinthetd, l1 + l2)

    A, B = complex(0), complex(0)
    for ld in range(abs(l1 - l2), l1 + l2 + 1):
        a5, b5 = ab5_coefficients(l1, m1, l2, m2, ld)
        A += a5 * sph_hankel[ld] * legendre[ld][abs(m1 - m2)]
        B += b5 * sph_hankel[ld] * legendre[ld][abs(m1 - m2)]
    A, B = eimph * A, eimph * B
    return A, B


def translation_coefficients_svwf_out_to_out(l1, m1, l2, m2, k, d, sph_bessel=None, legendre=None, exp_immphi=None):
    """Return the coefficients of the translation operator for the expansion of an outgoing spherical wave in terms of
    outgoing spherical waves with respect to a different origin.
    The output is a tuple (A,B), where the translation operator
        trans = \delta_pp' * A + (1-\delta_pp') * B

    Input:
    l1          integaer: l=1,...: Original wave's SVWF multipole degree
    m1          integaer: m=-l,...,l: Original wave's SVWF multipole order
    l2          integaer: l=1,...: Partial wave's SVWF multipole degree
    m2          integaer: m=-l,...,l: Partial wave's SVWF multipole order
    k           complex: wavenumber (inverse length unit)
    d           translation vectors in format [dx, dy, dz] (length unit)
                dx, dy, dz can be scalars or ndarrays
    sph_bessel  list: sph_bessel[i] contains the spherical bessel funciton of degree i, evaluated at k*d where d is the
                norm of the distance vector(s)
    legendre    list of lists: legendre[l][m] contains the legendre function of order l and degree m, evaluated at
                cos(theta) where theta is the polar angle(s) of the distance vector(s)
    """
    # spherical coordinates of d:
    dd = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

    if exp_immphi is None:
        phid = np.arctan2(d[1], d[0])
        eimph = np.exp(1j * (m1 - m2) * phid)
    else:
        eimph = exp_immphi[m1][m2]

    if sph_bessel is None:
        sph_bessel = [sf.spherical_bessel(n, k * dd) for n in range(l1 + l2 + 1)]

    if legendre is None:
        costthetd = d[2] / dd
        sinthetd = np.sqrt(d[0] ** 2 + d[1] ** 2) / dd
        legendre, _, _ = sf.legendre_normalized(costthetd, sinthetd, l1 + l2)

    A, B = complex(0), complex(0)
    for ld in range(abs(l1 - l2), l1 + l2 + 1):
        a5, b5 = ab5_coefficients(l1, m1, l2, m2, ld)
        A += a5 * sph_bessel[ld] * legendre[ld][abs(m1 - m2)]
        B += b5 * sph_bessel[ld] * legendre[ld][abs(m1 - m2)]
    A, B = eimph * A, eimph * B
    return A, B



def ab5_coefficients(l1, m1, l2, m2, p, symbolic=False):
    """Return a tuple (a5, b5) where a5 and b5 are the coefficients used in the evaluation of the SVWF translation
    operator. The computation is based on the sympy.physics.wigner package and is performed with symbolic numbers.
    If symbolic=True is specified as input argument, symbolic numbers are returned. Otherwise, complex (default).
    """
    jfac = sympy.I ** (abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * (-1) ** (m1 - m2)
    fac1 = sympy.sqrt((2 * l1 + 1) * (2 * l2 + 1) / sympy.S(2 * l1 * (l1 + 1) * l2 * (l2 + 1)))
    fac2a = (l1 * (l1 + 1) + l2 * (l2 + 1) - p * (p + 1)) * sympy.sqrt(2 * p + 1)
    fac2b = sympy.sqrt((l1 + l2 + 1 + p) * (l1 + l2 + 1 - p) * (p + l1 - l2) * (p - l1 + l2) * (2 * p + 1))
    wig1 = sympy.physics.wigner.wigner_3j(l1, l2, p, m1, -m2, -(m1 - m2))
    wig2a = sympy.physics.wigner.wigner_3j(l1, l2, p, 0, 0, 0)
    wig2b = sympy.physics.wigner.wigner_3j(l1, l2, p - 1, 0, 0, 0)

    if symbolic:
        a = jfac * fac1 * fac2a * wig1 * wig2a
        b = jfac * fac1 * fac2b * wig1 * wig2b
    else:
        a = complex(jfac * fac1 * fac2a * wig1 * wig2a)
        b = complex(jfac * fac1 * fac2b * wig1 * wig2b)
    return a, b


