# -*- coding: utf-8 -*-
"""Test the functions defined in layerresponse.py and layerresponse_mp.py."""

import unittest
import numpy as np
import smuthi.layerresponse
import smuthi.layerresponse_mp


class LayerResponseTest(unittest.TestCase):
    def setUp(self):
        self.layer_d = [0, 300, 400, 0]
        self.layer_n = [1, 2 + 0.1j, 3, 1 + 5j]
        self.omega = 2 * 3.15 / 550
        self.kpar = self.omega * 1.7
        self.precision = 15

    def test_layerresponse_mpmath_equals_numpy(self):
        """Are the results from layerresponse.py and from layerresponse_mp.py consistent?"""
        for pol in [0, 1]:
            for fromlayer in range(len(self.layer_d)):
                for tolayer in range(len(self.layer_d)):
                    lmat1 = smuthi.layerresponse.layersystem_response_matrix(pol, self.layer_d, self.layer_n, self.kpar,
                                                                      self.omega, fromlayer, tolayer)
                    lmat2 = smuthi.layerresponse_mp.layersystem_response_matrix(pol, self.layer_d, self.layer_n, self.kpar,
                                                                         self.omega, fromlayer, tolayer, self.precision)
                    np.testing.assert_almost_equal(lmat1, lmat2)

    def test_scattering_matrix_equals_transfer_matrix(self):
        """Are the results from the transfer matrix algorithm and from the scattering matrix algorithm consistent?"""
        for pol in [0, 1]:
            tmat = smuthi.layerresponse_mp.layersystem_transfer_matrix(pol, self.layer_d, self.layer_n, self.kpar, self.omega)
            smat = smuthi.layerresponse_mp.layersystem_scattering_matrix(pol, self.layer_d, self.layer_n, self.kpar,
                                                                  self.omega)
            self.assertAlmostEqual(tmat[1, 0] / tmat[0, 0], smat[1, 0])

    def test_layerresponse_equals_TSPL(self):
        """Are the results from layerresponse.py and consistent with the MATLAB prototype code TSPL?"""
        pol = 0
        fromlayer = 2
        tolayer = 1
        lmat = smuthi.layerresponse.layersystem_response_matrix(pol, self.layer_d, self.layer_n, self.kpar, self.omega,
                                                         fromlayer, tolayer)
        lmat_TSPL = np.array([[-0.392979481352895 - 0.376963315605839j, -0.455367266697897 + 0.426065579868901j],
                              [0.545168303416962 - 0.345873455516963j, -0.361796569025878 - 0.644799225334747j]])
        np.testing.assert_almost_equal(lmat, lmat_TSPL)

        pol = 1
        fromlayer = 1
        tolayer = 2
        lmat = smuthi.layerresponse.layersystem_response_matrix(pol, self.layer_d, self.layer_n, self.kpar, self.omega,
                                                         fromlayer, tolayer)
        lmat_TSPL = np.array([[-0.240373686730040 - 0.148769054113797j, 0.161922209423045 + 0.222085165907288j],
                              [-0.182951011363592 + 0.138158890222525j, 0.215395950986834 - 0.057346289106977j]])
        np.testing.assert_almost_equal(lmat, lmat_TSPL)


if __name__ == '__main__':
    unittest.main()
