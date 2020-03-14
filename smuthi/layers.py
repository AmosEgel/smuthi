# -*- coding: utf-8 -*-
"""Provide class for the representation of planar layer systems."""

import numpy as np
import mpmath
import smuthi.utility.memoizing as memo
import smuthi.fields.expansions as fldex
import smuthi.fields as flds


# global variables
matrix_format = np.array
math_module = np
precision = None


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
        """Return total number of layers

        Returns:
            number of layers
        """
        return len(self.thicknesses)

    def lower_zlimit(self, i):
        """Return the z-coordinate of lower boundary

        The coordinate system is defined such that z=0 corresponds to the interface between layer 0 and layer 1.

        Args:
            i (int):      index of layer in question (must be between 0 and number_of_layers-1)

        Returns:
            z-coordinate of lower boundary
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

        Args:
            i (int):      index of layer in question (must be between 0 and number_of_layers-1)

        Returns:
            z-coordinate of upper boundary
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

        Args:
            i (int):      index of layer in question (must be between 0 and number_of_layers-1)

        Returns:
            anchor point's z-coordinate
        """
        if i == 0:
            return self.upper_zlimit(i)
        else:
            return self.lower_zlimit(i)

    def layer_number(self, z):
        """ Return number of layer that contains point [0,0,z]

        If z is on the interface, the higher layer number is selected.

        Args:
            z (float):       z-coordinate of query point (length unit)

        Returns:
            number of layer containing z
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
        """Evaluate the layer system response to an electromagnetic excitation inside the layer system.

        Args:
            pwe (tuple or smuthi.field_expansion.PlaneWaveExpansion):  Either specify a PlaneWaveExpansion object that
                                                                       that represents the electromagnetic excitation,
                                                                       or a tuple of two PlaneWaveExpansion objects
                                                                       representing the upwards- and downwards
                                                                       propagating partial waves of the excitation.
            from_layer (int):   Layer number in which the excitation is located
            to_layer (int):     Layer number in which the layer response is to be evaluated

        Returns:
            Tuple (pwe_up, pwe_sown) of PlaneWaveExpansion objects representing the layer system response to the
            excitation.
        """
        if hasattr(pwe, '__len__'):
            if len(pwe) == 2:
                pwe_up_0, pwe_down_0 = self.response(pwe[0], from_layer, to_layer)
                pwe_up_1, pwe_down_1 = self.response(pwe[1], from_layer, to_layer)
                pwe_up = pwe_up_0 + pwe_up_1
                pwe_down = pwe_down_0 + pwe_down_1
            else:
                raise ValueError('pwe argument must be either PlaneWaveExpansion or tuple of length two')
        else:
            assert pwe.reference_point == [0, 0, self.reference_z(from_layer)]
            omega = pwe.k / self.refractive_indices[from_layer]
            k_to_layer = omega * self.refractive_indices[to_layer]
            reference_point = [0, 0, self.reference_z(to_layer)]
            loz, upz = self.lower_zlimit(to_layer), self.upper_zlimit(to_layer)
            pwe_up = fldex.PlaneWaveExpansion(k=k_to_layer, k_parallel=pwe.k_parallel,
                                              azimuthal_angles=pwe.azimuthal_angles,
                                              kind='upgoing', reference_point=reference_point,
                                              lower_z=loz, upper_z=upz)
            pwe_down = fldex.PlaneWaveExpansion(k=k_to_layer, k_parallel=pwe.k_parallel,
                                                azimuthal_angles=pwe.azimuthal_angles,
                                                kind='downgoing', reference_point=reference_point,
                                                lower_z=loz, upper_z=upz)
            for pol in range(2):
                L = layersystem_response_matrix(pol, self.thicknesses, self.refractive_indices, pwe.k_parallel, omega,
                                                from_layer, to_layer)
                if pwe.kind == 'upgoing':
                    pwe_up.coefficients[pol, :, :] = L[0, 0, :][:, None] * pwe.coefficients[pol, :, :]
                    pwe_down.coefficients[pol, :, :] = L[1, 0, :][:, None] * pwe.coefficients[pol, :, :]
                elif pwe.kind == 'downgoing':
                    pwe_up.coefficients[pol, :, :] = L[0, 1, :][:, None] * pwe.coefficients[pol, :, :]
                    pwe_down.coefficients[pol, :, :] = L[1, 1, :][:, None] * pwe.coefficients[pol, :, :]
                else:
                    raise ValueError('pwe type undefined')

        return pwe_up, pwe_down

    def wavenumber(self, layer_number, vacuum_wavelength):
        """
        Args:
            layer_number (int): number of layer in question
            vacuum_wavelength (float): vacuum wavelength

        Returns:
            wavenumber in that layer as float
        """
        return self.refractive_indices[layer_number] * flds.angular_frequency(vacuum_wavelength=vacuum_wavelength)


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
        Interface transition matrix as 2x2 numpy array or as 2x2 mpmath.matrix
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
        Layer propagation matrix as 2x2 numpy array or as 2x2 mpmath.matrix
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
        Transfer matrix as 2x2 numpy array or as 2x2 mpmath.matrix
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
        Scattering matrix as 2x2 numpy array or as 2x2 mpmath.matrix
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


@memo.Memoize
def layersystem_response_matrix(pol, layer_d, layer_n, kpar, omega, fromlayer, tolayer, prec=None):
    """Layer system response matrix of a planarly layered medium.

    Args:
        pol (int):                          polarization(0=TE, 1=TM)
        layer_d (list):                     layer thicknesses
        layer_n (list):                     complex layer refractive indices
        kpar (float or array like or str):  in-plane wavenumber.
        omega (float):                      angular frequency in units of c=1: omega=2*pi/lambda
        fromlayer (int):                    number of layer where the excitation is located
        tolayer (int):                      number of layer where the response is evaluated
        prec (int or None):                 allows to set the precision to this value (see set_precision)

    Returns:
        Layer system response matrix as a 2x2 array if kpar is float, or as 2x2xN array if kpar is array with len = N.
    """
    if not prec == precision:
        set_precision(prec)
    
    if hasattr(kpar, "__len__"):    # is kpar an array? then use recursive call to fill an 2 x 2 x N ndarray
        result = np.zeros((2, 2, len(kpar)), dtype=complex)
        for i, kp in enumerate(kpar):
            result[:, :, i] = layersystem_response_matrix(pol, layer_d, layer_n, kp, omega, fromlayer, tolayer, prec)
        return result

    if fromlayer == 0:  # bottom excitation
        smat = layersystem_scattering_matrix(pol, layer_d, layer_n, kpar, omega)
        lmat = matrix_format([[0, 0], [smat[1, 0], 0]])
    elif fromlayer == len(layer_d)-1:  # top excitation
        smat = layersystem_scattering_matrix(pol, layer_d, layer_n, kpar, omega)
        lmat = matrix_format([[0, smat[0, 1]], [0, 0]])
    else:  # excitation from inside
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
    """
    Args:
        m1 (mpmath.matrix or numpy.ndarray):    first matrix
        m2 (mpmath.matrix or numpy.ndarray):    second matrix

    Returns:
        matrix product m1 * m2 with same data type as m1 and m2
    """
    if isinstance(m1, mpmath.matrix) and isinstance(m2, mpmath.matrix):
        return m1 * m2
    elif isinstance(m1, np.ndarray) and isinstance(m2, np.ndarray):
        return np.dot(m1, m2)


def matrix_inverse(m):
    """
    Args:
        m (mpmath.matrix or numpy.ndarray):    matrix to invert

    Returns:
        inverse of m with same data type as m1 and m2
    """
    if isinstance(m, mpmath.matrix):
        return m ** (-1)
    elif isinstance(m, np.ndarray):
        return np.linalg.inv(m)


def set_precision(prec=None):
    """Set the numerical precision of the layer system response. You can use this to evaluate the layer response of
    unstable systems, for example in the case of evanescent waves in very thick layers. Calculations take longer time if
    the precision is set to a value other than None (default).

    Args:
        prec (None or int): If None, calculations are done using standard double precision. If int, that many decimal
                            digits are considered in the calculations, using the mpmath package.
    """
    global matrix_format
    global math_module
    global precision

    precision = prec
    if prec is None:
        print('Setting precision to standard numpy')
        matrix_format = np.array
        math_module = np
    else:
        print('Setting precision ', prec, ' digits')
        mpmath.mp.dps = prec
        matrix_format = mpmath.matrix
        math_module = mpmath
