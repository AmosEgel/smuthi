# -*- coding: utf-8 -*-

import numpy as np
import sympy

def fresnel_r(pol, kz1, kz2, n1, n2):
    """Return the complex Fresnel reflection coefficient.

    Input:
    pol     polarization (0=TE, 1=TM)
    kz1     incoming wave's z-wavenumber (k*cos(alpha1))
    kz2     transmitted wave's z-wavenumber (k*cos(alpha2))
    n1      first medium's complex refractive index (n+ik)
    n2      second medium's complex refractive index (n+ik)
    """
    if pol == 0:
        return (kz1 - kz2) / (kz1 + kz2)
    else:
        return (n2 ** 2 * kz1 - n1 ** 2 * kz2) / (n2 ** 2 * kz1 + n1 ** 2 * kz2)


def fresnel_t(pol, kz1, kz2, n1, n2):
    """Return the complex Fresnel transmission coefficient.

    Input:
    pol     polarization (0=TE, 1=TM)
    kz1     incoming wave's z-wavenumber (k*cos(alpha1))
    kz2     transmitted wave's z-wavenumber (k*cos(alpha2))
    n1      first medium's complex refractive index (n+ik)
    n2      second medium's complex refractive index (n+ik)
    """
    if pol == 0:
        return 2 * kz1 / (kz1 + kz2)
    else:
        return 2 * n1 * n2 * kz1 / (n2 ** 2 * kz1 + n1 ** 2 * kz2)


def interface_transition_matrix(pol, kz1, kz2, n1, n2):
    """Return the interface transition matrix to be used in the Transfer matrix algorithm as sympy.mpmath matrix.

    Input:
    pol     polarization (0=TE, 1=TM)
    kz1     incoming wave's z-wavenumber (k*cos(alpha1))
    kz2     transmitted wave's z-wavenumber (k*cos(alpha2))
    n1      first medium's complex refractive index (n+ik)
    n2      second medium's complex refractive index (n+ik)
    """
    t = fresnel_t(pol, kz1, kz2, n1, n2)
    r = fresnel_r(pol, kz1, kz2, n1, n2)
    return 1 / t * sympy.mpmath.matrix([[1, r], [r, 1]])


def layer_propagation_matrix(kz, d):
    """Return the layer propagation matrix to be used in the Transfer matrix algorithm as sympy.mpmath matrix.

    Input:
    kz      z-wavenumber (k*cos(alpha))
    d       thickness of layer
    """
    return sympy.mpmath.matrix([[sympy.mpmath.exp(-1j * kz * d), 0], [0, sympy.mpmath.exp(1j * kz * d)]])


def layersystem_transfer_matrix(pol, layer_d, layer_n, kpar, omega):
    """Return the transfer matrix of a planarly layered medium.

    Input:
    pol         polarization(0=TE, 1=TM)
    layer_d     numpy array of layer thicknesses
    layer_n     numpy array of complex layer refractive indices
    kpar        in-plane wavenumber
    omega       angular frequency in units of c=1: omega=2*pi/lambda
    """
    layer_kz = []
    for n in layer_n:
        kz = sympy.mpmath.sqrt((omega * n) ** 2 - kpar ** 2 + 0j)
        if kz.imag < 0:
            kz = -kz
        layer_kz.append(kz)
    tmat = sympy.mpmath.eye(2)
    for i in range(len(layer_d) - 1):
        dmat = interface_transition_matrix(pol, layer_kz[i], layer_kz[i + 1], layer_n[i], layer_n[i + 1])
        pmat = layer_propagation_matrix(layer_kz[i], layer_d[i])
        tmat = tmat * pmat * dmat
    return tmat


def layersystem_scattering_matrix(pol, layer_d, layer_n, kpar, omega):
    """Return the scattering matrix of a planarly layered medium.

    Input:
    pol         polarization(0=TE, 1=TM)
    layer_d     numpy array of layer thicknesses
    layer_n     numpy array of complex layer refractive indices
    kpar        in-plane wavenumber
    omega       angular frequency in units of c=1: omega=2*pi/lambda
    """
    layer_kz = []
    for n in layer_n:
        kz = sympy.mpmath.sqrt((omega * n) ** 2 - kpar ** 2 + 0j)
        if kz.imag < 0:
            kz = -kz
        layer_kz.append(kz)
    smat = sympy.mpmath.eye(2)
    for i in range(len(layer_d) - 1):
        dmat = interface_transition_matrix(pol, layer_kz[i], layer_kz[i + 1], layer_n[i], layer_n[i + 1])
        pmat = layer_propagation_matrix(layer_kz[i], layer_d[i])
        tmat = pmat * dmat
        s11 = smat[0, 0] / (tmat[0, 0] - smat[0, 1] * tmat[1, 0])
        s12 = (smat[0, 1] * tmat[1, 1] - tmat[0, 1]) / (tmat[0, 0] - smat[0, 1] * tmat[1, 0])
        s21 = smat[1, 1] * tmat[1, 0] * s11 + smat[1, 0]
        s22 = smat[1, 1] * tmat[1, 0] * s12 + smat[1, 1] * tmat[1, 1]
        smat = sympy.mpmath.matrix([[s11, s12], [s21, s22]])
    return smat


def layersystem_response_matrix(pol, layer_d, layer_n, kpar, omega, fromlayer, tolayer, precision):
    """Return the layer system response matrix of a planarly layered medium.

    Input:
    pol         polarization(0=TE, 1=TM)
    layer_d     list of layer thicknesses
    layer_n     list of complex layer refractive indices
    kpar        in-plane wavenumber
    omega       angular frequency in units of c=1: omega=2*pi/lambda
    fromlayer   number of layer where the excitation is located
    tolayer     number of layer where the response is evaluated
    precision   number of decimal digits to keep in interim calculations
    """
    sympy.mpmath.mp.dps = precision
    layer_d_above = [0] + layer_d[fromlayer:]
    layer_n_above = [layer_n[fromlayer]] + layer_n[fromlayer:]
    smat_above = layersystem_scattering_matrix(pol, layer_d_above, layer_n_above, kpar, omega)
    layer_d_below = layer_d[: fromlayer] + [0]
    layer_n_below = layer_n[: fromlayer] + [layer_n[fromlayer]]
    smat_below = layersystem_scattering_matrix(pol, layer_d_below, layer_n_below, kpar, omega)
    lmat = ((sympy.mpmath.matrix([[1, -smat_below[0, 1]], [-smat_above[1, 0], 1]]) ** -1)
            * sympy.mpmath.matrix([[0, smat_below[0, 1]], [smat_above[1, 0], 0]]))

    if tolayer > fromlayer:
        tmat_fromto = layersystem_transfer_matrix(pol, layer_d[fromlayer:tolayer + 1], layer_n[fromlayer:tolayer + 1],
                                                  kpar, omega)
        lmat = (tmat_fromto ** -1) * (lmat + sympy.mpmath.matrix([[1, 0], [0, 0]]))
    elif tolayer < fromlayer:
        tmat_fromto = layersystem_transfer_matrix(pol, layer_d[tolayer:fromlayer + 1], layer_n[tolayer:fromlayer + 1],
                                              kpar, omega)
        lmat = tmat_fromto * (lmat + sympy.mpmath.matrix([[0, 0], [0, 1]]))

    return np.array(lmat.tolist(), dtype=complex)
