# -*- coding: utf-8 -*-
"""Test the layerresponse functions defined in layers.py."""

import unittest
import numpy as np
import smuthi.layers as lay
import smuthi.field_expansion as fldex


layer_d = [0, 300, 400, 0]
layer_n = [1, 2 + 0.1j, 3, 1 + 5j]
omega = 2 * 3.15 / 550
kpar = omega * 1.7
precision = 15


def test_layerresponse_mpmath_equals_numpy():
    """Are the results with multiple precision consistent with numpy equivalent?"""
    for pol in [0, 1]:
        for fromlayer in range(len(layer_d)):
            for tolayer in range(len(layer_d)):
                lay.set_precision(precision=None)
                lmat1 = lay.layersystem_response_matrix(pol, layer_d, layer_n, kpar, omega, fromlayer, tolayer)
                lay.set_precision(precision=precision)
                lmat2 = lay.layersystem_response_matrix(pol, layer_d, layer_n, kpar, omega, fromlayer, tolayer)
                np.testing.assert_almost_equal(lmat1, lmat2)
    lay.set_precision(precision=None)


def test_scattering_matrix_equals_transfer_matrix():
    """Are the results from the transfer matrix algorithm and from the scattering matrix algorithm consistent?"""
    for pol in [0, 1]:
        tmat = lay.layersystem_transfer_matrix(pol, layer_d, layer_n, kpar, omega)
        smat = lay.layersystem_scattering_matrix(pol, layer_d, layer_n, kpar, omega)
        np.testing.assert_almost_equal(tmat[1, 0] / tmat[0, 0], smat[1, 0])


def test_layerresponse_against_prototype():
    """Are the results from layers.py and consistent with the MATLAB prototype code TSPL?"""
    pol = 0
    fromlayer = 2
    tolayer = 1
    lmat = lay.layersystem_response_matrix(pol, layer_d, layer_n, kpar, omega, fromlayer, tolayer)
    lmat_TSPL = np.array([[-0.392979481352895 - 0.376963315605839j, -0.455367266697897 + 0.426065579868901j],
                          [0.545168303416962 - 0.345873455516963j, -0.361796569025878 - 0.644799225334747j]])
    np.testing.assert_almost_equal(lmat, lmat_TSPL)

    pol = 1
    fromlayer = 1
    tolayer = 2
    lmat = lay.layersystem_response_matrix(pol, layer_d, layer_n, kpar, omega, fromlayer, tolayer)
    lmat_TSPL = np.array([[-0.240373686730040 - 0.148769054113797j, 0.161922209423045 + 0.222085165907288j],
                          [-0.182951011363592 + 0.138158890222525j, 0.215395950986834 - 0.057346289106977j]])
    np.testing.assert_almost_equal(lmat, lmat_TSPL)


def test_layerresponse_for_kpar_arrays():
    pol = 1
    fromlayer = 2
    tolayer = 1
    kpar_array = np.linspace(0, kpar)
    lmat_vec = lay.layersystem_response_matrix(pol, layer_d, layer_n, kpar_array, omega, fromlayer, tolayer)
    lmat_end = lay.layersystem_response_matrix(pol, layer_d, layer_n, kpar_array[-1], omega, fromlayer, tolayer)
    lmat0 = lay.layersystem_response_matrix(pol, layer_d, layer_n, kpar_array[0], omega, fromlayer, tolayer)
    np.testing.assert_almost_equal(lmat_end, lmat_vec[:, :, -1])
    np.testing.assert_almost_equal(lmat0, lmat_vec[:, :, 0])


def test_layerresponse_method():
    fromlayer=2
    tolayer=1
    kp = np.linspace(0, 2) * omega
    a = np.linspace(0, 2*np.pi)
    layer_system = lay.LayerSystem(thicknesses=layer_d, refractive_indices=layer_n)
    ref = [0, 0, layer_system.reference_z(fromlayer)]
    pwe_up = fldex.PlaneWaveExpansion(k=omega*1.2, k_parallel=kp, azimuthal_angles=a, type='upgoing',
                                        reference_point=ref)
    pwe_up.coefficients[0,:, :] = np.exp(-pwe_up.k_parallel_grid()/omega)
    pwe_down = fldex.PlaneWaveExpansion(k=omega * 1.2, k_parallel=kp, azimuthal_angles=a, type='downgoing',
                                          reference_point=ref)
    pwe_down.coefficients[0, :, :] = 2j * np.exp(-pwe_up.k_parallel_grid() / omega * 3)

    pwe_r_up, pwe_r_down = layer_system.response(pwe_up, fromlayer, tolayer)
    pwe_r_up2, pwe_r_down2 = layer_system.response(pwe_down, fromlayer, tolayer)
    pwe_r_up3, pwe_r_down3 = layer_system.response((pwe_up, pwe_down), fromlayer, tolayer)

    print(pwe_r_up.coefficients[0, 0, 0] + pwe_r_up2.coefficients[0, 0, 0])
    print(pwe_r_up3.coefficients[0, 0, 0])



if __name__ == '__main__':
    test_layerresponse_mpmath_equals_numpy()
    test_scattering_matrix_equals_transfer_matrix()
    test_layerresponse_against_prototype()
    test_layerresponse_for_kpar_arrays()
    test_layerresponse_method()
