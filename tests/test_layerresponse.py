# -*- coding: utf-8 -*-
"""Test the layerresponse functions defined in layers.py."""

import unittest
import numpy as np
import smuthi.layers


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
                lmat1 = smuthi.layers.layersystem_response_matrix(pol, layer_d, layer_n, kpar,
                                                                  omega, fromlayer, tolayer)
                lmat2 = smuthi.layers.layersystem_response_matrix(pol, layer_d, layer_n, kpar,
                                                                  omega, fromlayer, tolayer, precision)
                np.testing.assert_almost_equal(lmat1, lmat2)


def test_scattering_matrix_equals_transfer_matrix():
    """Are the results from the transfer matrix algorithm and from the scattering matrix algorithm consistent?"""
    for pol in [0, 1]:
        tmat = smuthi.layers.layersystem_transfer_matrix(pol, layer_d, layer_n, kpar, omega)
        smat = smuthi.layers.layersystem_scattering_matrix(pol, layer_d, layer_n, kpar, omega)
        np.testing.assert_almost_equal(tmat[1, 0] / tmat[0, 0], smat[1, 0])


def test_layerresponse_equals_TSPL():
    """Are the results from layers.py and consistent with the MATLAB prototype code TSPL?"""
    pol = 0
    fromlayer = 2
    tolayer = 1
    lmat = smuthi.layers.layersystem_response_matrix(pol, layer_d, layer_n, kpar, omega,
                                                     fromlayer, tolayer)
    lmat_TSPL = np.array([[-0.392979481352895 - 0.376963315605839j, -0.455367266697897 + 0.426065579868901j],
                          [0.545168303416962 - 0.345873455516963j, -0.361796569025878 - 0.644799225334747j]])
    np.testing.assert_almost_equal(lmat, lmat_TSPL)

    pol = 1
    fromlayer = 1
    tolayer = 2
    lmat = smuthi.layers.layersystem_response_matrix(pol, layer_d, layer_n, kpar, omega,
                                                     fromlayer, tolayer)
    lmat_TSPL = np.array([[-0.240373686730040 - 0.148769054113797j, 0.161922209423045 + 0.222085165907288j],
                          [-0.182951011363592 + 0.138158890222525j, 0.215395950986834 - 0.057346289106977j]])
    np.testing.assert_almost_equal(lmat, lmat_TSPL)


def test_layerresponse_for_kpar_arrays():
    pol = 1
    fromlayer = 2
    tolayer = 1
    kpar_array = np.linspace(0, kpar)
    lmat_vec = smuthi.layers.layersystem_response_matrix(pol, layer_d, layer_n, kpar_array, omega,
                                                         fromlayer, tolayer)
    lmat_end = smuthi.layers.layersystem_response_matrix(pol, layer_d, layer_n, kpar_array[-1], omega,
                                                     fromlayer, tolayer)
    lmat0 = smuthi.layers.layersystem_response_matrix(pol, layer_d, layer_n, kpar_array[0], omega,
                                                     fromlayer, tolayer)

    np.testing.assert_almost_equal(lmat_end, lmat_vec[:, :, -1])
    np.testing.assert_almost_equal(lmat0, lmat_vec[:, :, 0])


def test_layerresponse_lookup():
    trivial_layer_d = [0, 0]
    trivial_layer_n = [1, 1]
    kpar_array = np.linspace(0, kpar)

    fromlayer = 0
    tolayer = 0
    L1 = smuthi.layers.evaluate_layerresponse_lookup(trivial_layer_d, trivial_layer_n, kpar_array, omega, fromlayer,
                                                     tolayer, None)

    L1TE = smuthi.layers.layersystem_response_matrix(0, trivial_layer_d, trivial_layer_n, kpar_array, omega, fromlayer,
                                                     tolayer)

    L1TM = smuthi.layers.layersystem_response_matrix(1, trivial_layer_d, trivial_layer_n, kpar_array, omega, fromlayer,
                                                     tolayer)
    np.testing.assert_almost_equal(L1[0, :, :, :], L1TE)
    np.testing.assert_almost_equal(L1[1, :, :, :], L1TM)

    fromlayer = 0
    tolayer = 1
    L2 = smuthi.layers.evaluate_layerresponse_lookup(trivial_layer_d, trivial_layer_n, kpar_array, omega, fromlayer,
                                                     tolayer, None)
    L2TE = smuthi.layers.layersystem_response_matrix(0, trivial_layer_d, trivial_layer_n, kpar_array, omega, fromlayer,
                                                     tolayer)

    L2TM = smuthi.layers.layersystem_response_matrix(1, trivial_layer_d, trivial_layer_n, kpar_array, omega, fromlayer,
                                                     tolayer)

    np.testing.assert_almost_equal(L2[0, :, :, :], L2TE)
    np.testing.assert_almost_equal(L2[1, :, :, :], L2TM)

    fromlayer = 0
    tolayer = 0
    L1 = smuthi.layers.evaluate_layerresponse_lookup(trivial_layer_d, trivial_layer_n, kpar_array, omega, fromlayer,
                                                     tolayer, None)

    L1TE = smuthi.layers.layersystem_response_matrix(0, trivial_layer_d, trivial_layer_n, kpar_array, omega, fromlayer,
                                                     tolayer)

    L1TM = smuthi.layers.layersystem_response_matrix(1, trivial_layer_d, trivial_layer_n, kpar_array, omega, fromlayer,
                                                     tolayer)
    np.testing.assert_almost_equal(L1[0, :, :, :], L1TE)
    np.testing.assert_almost_equal(L1[1, :, :, :], L1TM)

    fromlayer = 0
    tolayer = 1
    L1 = smuthi.layers.evaluate_layerresponse_lookup(layer_d, layer_n, kpar_array, omega, fromlayer, tolayer, None)

    L1TE = smuthi.layers.layersystem_response_matrix(0, layer_d, layer_n, kpar_array, omega, fromlayer, tolayer)

    L1TM = smuthi.layers.layersystem_response_matrix(1, layer_d, layer_n, kpar_array, omega, fromlayer, tolayer)
    np.testing.assert_almost_equal(L1[0, :, :, :], L1TE)
    np.testing.assert_almost_equal(L1[1, :, :, :], L1TM)

if __name__ == '__main__':
    test_layerresponse_mpmath_equals_numpy()
    test_scattering_matrix_equals_transfer_matrix()
    test_layerresponse_equals_TSPL()
    test_layerresponse_for_kpar_arrays()
    test_layerresponse_lookup()

