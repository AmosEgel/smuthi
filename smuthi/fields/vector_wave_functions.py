# -*- coding: utf-8 -*-
"""This module contains the vector wave functions and their transformations."""
 
import numpy as np
import smuthi.utility.math as sf


def plane_vector_wave_function(x, y, z, kp, alpha, kz, pol):
    r"""Electric field components of plane wave (PVWF), see section 2.3.1 of
    [Egel 2018 diss].

    .. math::
        \mathbf{\Phi}_j = \exp ( \mathrm{i} \mathbf{k} \cdot \mathbf{r} ) \hat{ \mathbf{e} }_j

    with :math:`\hat{\mathbf{e}}_0` denoting the unit vector in azimuthal direction ('TE' or 's' polarization),
    and :math:`\hat{\mathbf{e}}_1` denoting the unit vector in polar direction ('TM' or 'p' polarization).

    The input arrays should have one of the following dimensions:

        - x,y,z: (N x 1) matrix
        - kp,alpha,kz: (1 x M) matrix
        - Ex, Ey, Ez: (M x N) matrix

    or

        - x,y,z: (M x N) matrix
        - kp,alpha,kz: scalar
        - Ex, Ey, Ez: (M x N) matrix

    Args:
        x (numpy.ndarray): x-coordinate of position where to test the field (length unit)
        y (numpy.ndarray): y-coordinate of position where to test the field
        z (numpy.ndarray): z-coordinate of position where to test the field
        kp (numpy.ndarray): parallel component of k-vector (inverse length unit)
        alpha (numpy.ndarray): azimthal angle of k-vector (rad)
        kz (numpy.ndarray): z-component of k-vector (inverse length unit)
        pol (int): Polarization (0=TE, 1=TM)

    Returns:
        - x-coordinate of PVWF electric field (numpy.ndarray)
        - y-coordinate of PVWF electric field (numpy.ndarray)
        - z-coordinate of PVWF electric field (numpy.ndarray)
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
    """Electric field components of spherical vector wave function (SVWF). The conventions are chosen according to
    `A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006
    <https://doi.org/10.1007/978-3-540-33697-6>`_
    See also section 2.3.2 of [Egel 2018 diss].

    Args:
        x (numpy.ndarray):      x-coordinate of position where to test the field (length unit)
        y (numpy.ndarray):      y-coordinate of position where to test the field
        z (numpy.ndarray):      z-coordinate of position where to test the field
        k (float or complex):   wavenumber (inverse length unit)
        nu (int):               1 for regular waves, 3 for outgoing waves
        tau (int):              spherical polarization, 0 for spherical TE and 1 for spherical TM
        l (int):                l=1,... multipole degree (polar quantum number)
        m (int):                m=-l,...,l multipole order (azimuthal quantum number)

    Returns:
        - x-coordinate of SVWF electric field (numpy.ndarray)
        - y-coordinate of SVWF electric field (numpy.ndarray)
        - z-coordinate of SVWF electric field (numpy.ndarray)
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    r = np.sqrt(x**2 + y**2 + z**2)
    non0 = np.logical_not(r == 0)
    
    theta = np.zeros(x.shape)
    theta[non0] = np.arccos(z[non0] / r[non0])
    phi = np.arctan2(y, x)

    # unit vector in r-direction

    er_x = np.zeros(x.shape)
    er_y = np.zeros(x.shape)
    er_z = np.ones(x.shape)

    er_x[non0] = x[non0] / r[non0]
    er_y[non0] = y[non0] / r[non0]
    er_z[non0] = z[non0] / r[non0]

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
        bes_kr = np.zeros(kr.shape, dtype=np.complex)
        dxxz_kr = np.zeros(kr.shape, dtype=np.complex)
        zero = (r==0)
        nonzero = np.logical_not(zero)
        bes_kr[zero] = (l == 1) / 3
        dxxz_kr[zero] = (l == 1) * 2 / 3
        bes_kr[nonzero] = (bes[nonzero] / kr[nonzero])
        dxxz_kr[nonzero] = (dxxz[nonzero] / kr[nonzero])
    elif nu == 3:
        bes = sf.spherical_hankel(l, kr)
        dxxz = sf.dx_xh(l, kr)
        bes_kr = bes / kr
        dxxz_kr = dxxz / kr
    else:
        raise ValueError('nu must be 1 (regular SVWF) or 3 (outgoing SVWF)')

    eimphi = np.exp(1j * m * phi)
    prefac = 1 / np.sqrt(2 * l * (l + 1))
    if tau == 0:
        Ex = prefac * bes * (1j * m * pilm * eth_x - taulm * eph_x) * eimphi
        Ey = prefac * bes * (1j * m * pilm * eth_y - taulm * eph_y) * eimphi
        Ez = prefac * bes * (1j * m * pilm * eth_z - taulm * eph_z) * eimphi
    elif tau == 1:
        Ex = prefac * (l * (l + 1) * bes_kr * plm * er_x + dxxz_kr * (taulm * eth_x + 1j * m * pilm * eph_x)) * eimphi
        Ey = prefac * (l * (l + 1) * bes_kr * plm * er_y + dxxz_kr * (taulm * eth_y + 1j * m * pilm * eph_y)) * eimphi
        Ez = prefac * (l * (l + 1) * bes_kr * plm * er_z + dxxz_kr * (taulm * eth_z + 1j * m * pilm * eph_z)) * eimphi
    else:
        raise ValueError('tau must be 0 (spherical TE) or 1 (spherical TM)')

    return Ex, Ey, Ez
