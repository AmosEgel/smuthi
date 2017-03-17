# -*- coding: utf-8 -*-

import numpy as np
import smuthi.spherical_functions
import smuthi.index_conversion


def mie_coefficient(tau, l, k_medium, k_particle, radius):
    """Return the Mie coefficients of a sphere.

    Input:
    tau         integer: spherical polarization, 0 for spherical TE and 1 for spherical TM
    l           integer: l=1,... multipole degree (polar quantum number)
    k_medium    float or complex: wavenumber in surrounding medium (inverse length unit)
    k_particle  float or complex: wavenumber inside sphere (inverse length unit)
    radius      float: radius of sphere (length unit)
    """
    jlkr_medium = smuthi.spherical_functions.spherical_bessel(l, k_medium * radius)
    jlkr_particle = smuthi.spherical_functions.spherical_bessel(l, k_particle * radius)
    dxxj_medium = smuthi.spherical_functions.dx_xj(l, k_medium * radius)
    dxxj_particle = smuthi.spherical_functions.dx_xj(l, k_particle * radius)

    hlkr_medium = smuthi.spherical_functions.spherical_hankel(l, k_medium * radius)
    dxxh_medium = smuthi.spherical_functions.dx_xh(l, k_medium * radius)

    if tau == 0:
        q = (jlkr_medium * dxxj_particle - jlkr_particle * dxxj_medium) / (jlkr_particle * dxxh_medium - hlkr_medium *
                                                                           dxxj_particle)
    elif tau == 1:
        q = ((k_medium ** 2 * jlkr_medium * dxxj_particle - k_particle ** 2 * jlkr_particle * dxxj_medium) /
             (k_particle ** 2 * jlkr_particle * dxxh_medium - k_medium ** 2 * hlkr_medium * dxxj_particle))
    else:
        raise ValueError('tau must be 0 (spherical TE) or 1 (spherical TM)')

    return q

def t_matrix_sphere(k_medium, k_particle, radius, lmax, mmax=None, index_arrangement='stlm'):
    """Return the T-matrix of a spherical scattering object.

    Input:
    k_medium            float or complex: wavenumber in surrounding medium (inverse length unit)
    k_particle          float or complex: wavenumber inside sphere (inverse length unit)
    radius              float: radius of sphere (length unit)
    lmax:               truncation degree of SVWF expansion
    mmax:               (optional) truncation order of SVWF expansion, i.e., |m|<=mmax, default: mmax=lmax
    index_arrangement:  (optional) string to specify the order according to which the indices are arranged
                        See smuthi.index_conversion for explanation. Default: 'stlm'
    """
    nmax = smuthi.index_conversion.max_index(lmax, mmax, 1, index_arrangement)
    t = np.zeros((nmax+1, nmax+1), dtype=complex)
    for tau in range(2):
        for l in range(1, lmax+1):
            for m in range(-l, l + 1):
                n = smuthi.index_conversion.multi2single(tau, l, m, lmax, mmax, index_arrangement=index_arrangement)
                t[n, n] = mie_coefficient(tau, l, k_medium, k_particle, radius)

    return t