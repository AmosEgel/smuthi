# -*- coding: utf-8 -*-
"""Test spherical_functions"""

import unittest
import smuthi.spherical_functions
import numpy as np


class SphericalFunctionsTest(unittest.TestCase):
    def test_against_prototype(self):
        lmax = 3
        omega = 2 * 3.14 / 550
        kp = omega * np.array([0.01, 0.2, 0.7, 0.99, 1.2, 2 - 0.5j])
        kz = np.sqrt(omega ** 2 - kp ** 2 + 0j)
        kz[kz.imag < 0] = -kz[kz.imag < 0]
        ct = kz / omega
        st = kp / omega
        plm, pilm, taulm = smuthi.spherical_functions.legendre_normalized(ct, st, lmax)

        # P_3^0
        self.assertAlmostEqual(plm[3][0][0], 1.870267465826245)
        self.assertAlmostEqual(plm[3][0][1], 1.649727250184103)
        self.assertAlmostEqual(plm[3][0][2], -0.300608757357466)
        self.assertAlmostEqual(plm[3][0][3], -0.382739631607606)
        self.assertAlmostEqual(plm[3][0][4], - 3.226515147957620j)
        self.assertAlmostEqual(plm[3][0][5], -25.338383323084539 - 22.141864985871653j)

        # P_2^1
        self.assertAlmostEqual(plm[2][1][0], 0.019363948460993)
        self.assertAlmostEqual(plm[2][1][1], 0.379473319220206)
        self.assertAlmostEqual(plm[2][1][2], 0.968052168015753)
        self.assertAlmostEqual(plm[2][1][3], 0.270444009917026)
        self.assertAlmostEqual(plm[2][1][4], 1.541427909439815j)
        self.assertAlmostEqual(plm[2][1][5], 3.906499971729346 + 6.239600710712296j)

        # pi_2^1
        self.assertAlmostEqual(pilm[2][1][0], 1.936394846099318)
        self.assertAlmostEqual(pilm[2][1][1], 1.897366596101028)
        self.assertAlmostEqual(pilm[2][1][2], 1.382931668593933)
        self.assertAlmostEqual(pilm[2][1][3], 0.273175767592955)
        self.assertAlmostEqual(pilm[2][1][4], 1.284523257866512j)
        self.assertAlmostEqual(pilm[2][1][5], 1.104282256024128 + 3.395870919362181j)

        # tau_3^2
        self.assertAlmostEqual(taulm[3][2][0], 0.051227068616724)
        self.assertAlmostEqual(taulm[3][2][1], 0.963213372000203)
        self.assertAlmostEqual(taulm[3][2][2], 0.950404683542753)
        self.assertAlmostEqual(taulm[3][2][3], -2.384713931794872)
        self.assertAlmostEqual(taulm[3][2][4], -7.131877733107878)
        self.assertAlmostEqual(taulm[3][2][5], -39.706934218093430 + 42.588889121019569j)


if __name__ == '__main__':
    unittest.main()
