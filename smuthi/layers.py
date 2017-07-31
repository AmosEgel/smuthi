# -*- coding: utf-8 -*-
"""Provide class for the representation of planar layer systems."""

import numpy as np
import sympy
from functools import lru_cache
import smuthi.memoizing as memo
import smuthi.field_expansion as fldex


# global variables
matrix_format = np.array
math_module = np
layerresponse_lookup = {'args': {}, 'data': None}


class LayerSystem:
    """Stack of planar layers.

    Args:
        thicknesses (list):         layer thicknesses, first and last are semi inf and set to 0 (length unit)
        refractive_indices (list):  complex refractive indices in the form n+jk
    """
    def __init__(self, thicknesses=None, refractive_indices=None):
        if thicknesses is None:
            thicknesses = [0, 0]
        if refractive_indices is None:
            refractive_indices = [1, 1]
        self.thicknesses = thicknesses
        self.thicknesses[0] = 0
        self.thicknesses[-1] = 0
        self.refractive_indices = refractive_indices

    def number_of_layers(self):
        """Return total number of layers"""
        return len(self.thicknesses)

    def lower_zlimit(self, i):
        """Return the z-coordinate of lower boundary

        The coordinate system is defined such that z=0 corresponds to the interface between layer 0 and layer 1.

        input:
        i:      index of layer in question (must be between 0 and number_of_layers-1)
        """
        if i == 0:
            return -np.inf
        else:
            sumthick = 0
            for d in self.thicknesses[1:i]:
                sumthick += d
        return sumthick

    def upper_zlimit(self, i):
        """Return the z-coordinate of upper boundary.

        The coordinate system is defined such that z=0 corresponds to the interface between layer 0 and layer 1.

        input:
        i:      index of layer in question (must be between 0 and number_of_layers-1)
        """
        if i == self.number_of_layers() - 1:
            return np.inf
        else:
            sumthick = 0
            for d in self.thicknesses[1:i + 1]:
                sumthick += d
        return sumthick

    def reference_z(self, i):
        """Return the anchor point's z-coordinate.

        The coordinate system is defined such that z=0 corresponds to the interface between layer 0 and layer 1.

        input:
        i:      index of layer in question (must be between 0 and number_of_layers-1)
        """
        if i == 0:
            return self.upper_zlimit(i)
        else:
            return self.lower_zlimit(i)

    def layer_number(self, z):
        """ Return number of layer that contains point [0,0,z]

        If z is on the interface, the higher layer number is selected.

        input:
        z:       z-coordinate of query point (length unit)
        """
        d = 0
        laynum = 0
        for th in self.thicknesses[1:]:
            if z >= d:
                laynum += 1
                d += th
            else:
                break
        return laynum

    def response(self, pwe, from_layer, to_layer):

        # to do: check validity of pwe in from_layer, testing, docstring, check if reference point of pwe coincides with
        #        reference point of from_layer
        omega = pwe.k / self.refractive_indices[from_layer]
        k_to_layer = omega * self.refractive_indices[to_layer]
        reference_point = [0, 0, self.reference_z(to_layer)]
        valid_between = (self.lower_zlimit(to_layer), self.upper_zlimit(to_layer))
        pwe_up = fldex.PlaneWaveExpansion(k=k_to_layer, k_parallel=pwe.k_parallel,
                                          azimuthal_angles=pwe.azimuthal_angles,
                                          type='upgoing', reference_point=reference_point, valid_between=valid_between)
        pwe_down = fldex.PlaneWaveExpansion(k=k_to_layer, k_parallel=pwe.k_parallel,
                                            azimuthal_angles=pwe.azimuthal_angles,
                                            type='downgoing', reference_point=reference_point,
                                            valid_between=valid_between)
        for pol in range(2):
            L = layersystem_response_matrix(pol, self.thicknesses, self.refractive_indices, pwe.k_parallel, omega,
                                            from_layer, to_layer)
            if pwe.type == 'upgoing':
                pwe_up.coefficients[pol, :, :] = L[0, 0, :] * pwe.coefficients[pol, :, :]
                pwe_down.coefficients[pol, :, :] = L[1, 0, :] * pwe.coefficients[pol, :, :]
            elif pwe.type == 'downgoing':
                pwe_up.coefficients[pol, :, :] = L[0, 1, :] * pwe.coefficients[pol, :, :]
                pwe_down.coefficients[pol, :, :] = L[1, 1, :] * pwe.coefficients[pol, :, :]
            else:
                raise ValueError('pwe type undefined')

        return pwe_up, pwe_down


def fresnel_r(pol, kz1, kz2, n1, n2):
    """Fresnel reflection coefficient.

    Args:
        pol (int):              polarization (0=TE, 1=TM)
        kz1 (float or array):   incoming wave's z-wavenumber (k*cos(alpha1))
        kz2 (float or array):   transmitted wave's z-wavenumber (k*cos(alpha2))
        n1 (float or complex):  first medium's complex refractive index (n+ik)
        n2 (float or complex):  second medium's complex refractive index (n+ik)

    Returns:
        Complex Fresnel reflection coefficient (float or array)
    """
    if pol == 0:
        return (kz1 - kz2) / (kz1 + kz2)
    else:
        return (n2 ** 2 * kz1 - n1 ** 2 * kz2) / (n2 ** 2 * kz1 + n1 ** 2 * kz2)


def fresnel_t(pol, kz1, kz2, n1, n2):
    """Fresnel transmission coefficient.

    Args:
        pol (int):              polarization (0=TE, 1=TM)
        kz1 (float or array):   incoming wave's z-wavenumber (k*cos(alpha1))
        kz2 (float or array):   transmitted wave's z-wavenumber (k*cos(alpha2))
        n1 (float or complex):  first medium's complex refractive index (n+ik)
        n2 (float or complex):  second medium's complex refractive index (n+ik)

    Returns:
        Complex Fresnel transmission coefficient (float or array)
    """

    if pol == 0:
        return 2 * kz1 / (kz1 + kz2)
    else:
        return 2 * n1 * n2 * kz1 / (n2 ** 2 * kz1 + n1 ** 2 * kz2)


def interface_transition_matrix(pol, kz1, kz2, n1, n2):
    """Interface transition matrix to be used in the Transfer matrix algorithm.

    Args:
        pol (int):              polarization (0=TE, 1=TM)
        kz1 (float or array):   incoming wave's z-wavenumber (k*cos(alpha1))
        kz2 (float or array):   transmitted wave's z-wavenumber (k*cos(alpha2))
        n1 (float or complex):  first medium's complex refractive index (n+ik)
        n2 (float or complex):  second medium's complex refractive index (n+ik)

    Returns:
        Interface transition matrix as 2x2 numpy array or as 2x2 sympy.mpmath.matrix
    """
    t = fresnel_t(pol, kz1, kz2, n1, n2)
    r = fresnel_r(pol, kz1, kz2, n1, n2)
    return 1 / t * matrix_format([[1, r], [r, 1]])


def layer_propagation_matrix(kz, d):
    """Layer propagation matrix to be used in the Transfer matrix algorithm.

    Args:
        kz (float or complex):  z-wavenumber (k*cos(alpha))
        d  (float):             thickness of layer

    Returns:
        Layer propagation matrix as 2x2 numpy array or as 2x2 sympy.mpmath.matrix
    """
    return matrix_format([[math_module.exp(-1j * kz * d), 0], [0, math_module.exp(1j * kz * d)]])


def layersystem_transfer_matrix(pol, layer_d, layer_n, kpar, omega):
    """Transfer matrix of a planarly layered medium.

    Args:
        pol (int):      polarization(0=TE, 1=TM)
        layer_d (list): layer thicknesses
        layer_n (list): complex layer refractive indices
        kpar (float):   in-plane wavenumber
        omega (float):  angular frequency in units of c=1: omega=2*pi/lambda

    Returns:
        Transfer matrix as 2x2 numpy array or as 2x2 sympy.mpmath.matrix
    """
    layer_kz = []
    for n in layer_n:
        kz = math_module.sqrt((omega * n) ** 2 - kpar ** 2 + 0j)
        if kz.imag < 0:
            kz = -kz
        layer_kz.append(kz)
    tmat = math_module.eye(2)
    for i in range(len(layer_d) - 1):
        dmat = interface_transition_matrix(pol, layer_kz[i], layer_kz[i + 1], layer_n[i], layer_n[i + 1])
        pmat = layer_propagation_matrix(layer_kz[i], layer_d[i])
        tmat = matrix_product(tmat, matrix_product(pmat, dmat))
    return tmat


def layersystem_scattering_matrix(pol, layer_d, layer_n, kpar, omega):
    """Scattering matrix of a planarly layered medium.

    Args:
        pol (int):      polarization(0=TE, 1=TM)
        layer_d (list): layer thicknesses
        layer_n (list): complex layer refractive indices
        kpar (float):   in-plane wavenumber
        omega (float):  angular frequency in units of c=1: omega=2*pi/lambda

    Returns:
        Scattering matrix as 2x2 numpy array or as 2x2 sympy.mpmath.matrix
    """
    layer_kz = []
    for n in layer_n:
        kz = math_module.sqrt((omega * n) ** 2 - kpar ** 2 + 0j)
        if kz.imag < 0:
            kz = -kz
        layer_kz.append(kz)
    smat = math_module.eye(2)
    for i in range(len(layer_d) - 1):
        dmat = interface_transition_matrix(pol, layer_kz[i], layer_kz[i + 1], layer_n[i], layer_n[i + 1])
        pmat = layer_propagation_matrix(layer_kz[i], layer_d[i])
        tmat = matrix_product(pmat, dmat)
        s11 = smat[0, 0] / (tmat[0, 0] - smat[0, 1] * tmat[1, 0])
        s12 = (smat[0, 1] * tmat[1, 1] - tmat[0, 1]) / (tmat[0, 0] - smat[0, 1] * tmat[1, 0])
        s21 = smat[1, 1] * tmat[1, 0] * s11 + smat[1, 0]
        s22 = smat[1, 1] * tmat[1, 0] * s12 + smat[1, 1] * tmat[1, 1]
        smat = matrix_format([[s11, s12], [s21, s22]])
    return smat


#@lru_cache(128)
@memo.Memoize
def layersystem_response_matrix(pol, layer_d, layer_n, kpar, omega, fromlayer, tolayer):
    """Layer system response matrix of a planarly layered medium.

    Args:
        pol (int):                  polarization(0=TE, 1=TM)
        layer_d (list):             layer thicknesses
        layer_n (list):             complex layer refractive indices
        kpar (float or array like): in-plane wavenumber
        omega (float):              angular frequency in units of c=1: omega=2*pi/lambda
        fromlayer (int):            number of layer where the excitation is located
        tolayer (int):             number of layer where the response is evaluated

    Returns:
        Layer system response matrix as a 2x2 array if kpar is float, or as 2x2xN array if kpar is array with len = N.
    """

    if hasattr(kpar, "__len__"):    # is kpar an array? then use recursive call to fill an 2 x 2 x N ndarray
        result = np.zeros((2, 2, len(kpar)), dtype=complex)
        for i, kp in enumerate(kpar):
            result[:, :, i] = layersystem_response_matrix(pol, layer_d, layer_n, kp, omega, fromlayer, tolayer)
        return result

    layer_d_above = [0] + layer_d[fromlayer:]
    layer_n_above = [layer_n[fromlayer]] + layer_n[fromlayer:]
    smat_above = layersystem_scattering_matrix(pol, layer_d_above, layer_n_above, kpar, omega)
    layer_d_below = layer_d[: fromlayer] + [0]
    layer_n_below = layer_n[: fromlayer] + [layer_n[fromlayer]]
    smat_below = layersystem_scattering_matrix(pol, layer_d_below, layer_n_below, kpar, omega)
    lmat = matrix_product(matrix_inverse(matrix_format([[1, -smat_below[0, 1]], [-smat_above[1, 0], 1]])),
                          matrix_format([[0, smat_below[0, 1]], [smat_above[1, 0], 0]]))
    if tolayer > fromlayer:
        tmat_fromto = layersystem_transfer_matrix(pol, layer_d[fromlayer:tolayer + 1], layer_n[fromlayer:tolayer + 1],
                                                  kpar, omega)
        lmat = matrix_product(matrix_inverse(tmat_fromto), lmat + matrix_format([[1, 0], [0, 0]]))
    elif tolayer < fromlayer:
        tmat_fromto = layersystem_transfer_matrix(pol, layer_d[tolayer:fromlayer + 1], layer_n[tolayer:fromlayer + 1],
                                                  kpar, omega)
        lmat = matrix_product(tmat_fromto, lmat + matrix_format([[0, 0], [0, 1]]))
    return np.array(lmat.tolist(), dtype=complex)


def matrix_product(m1, m2):
    if isinstance(m1, sympy.mpmath.matrix) and isinstance(m2, sympy.mpmath.matrix):
        return m1 * m2
    elif isinstance(m1, np.ndarray) and isinstance(m2, np.ndarray):
        return np.dot(m1, m2)


def matrix_inverse(m):
    if isinstance(m, sympy.mpmath.matrix):
        return m ** (-1)
    elif isinstance(m, np.ndarray):
        return np.linalg.inv(m)


def set_precision(precision=None):
    """Set the numerical precision of the layer system response. You can use this to evaluate the layer response of
    unstable systems, for example in the case of evanescent waves in very thick layers. Calculations take longer time if
    the precision is set to a value other than None (default).

    Args:
        precision (None or int):    If None, calculations are done using standard double precision. If int, that many
                                    decimal digits are considered in the calculations, using the mpmath package.
    """
    global matrix_format
    global math_module

    if precision is None:
        matrix_format = np.array
        math_module = np
    else:
        sympy.mpmath.mp.dps = precision
        matrix_format = sympy.mpmath.matrix
        math_module = sympy.mpmath
