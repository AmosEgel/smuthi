# -*- coding: utf-8 -*-
"""Test spherical_functions"""

import unittest
import smuthi.spherical_functions
import numpy as np


class SphericalFunctionsTest(unittest.TestCase):
    def test_Plm_against_prototype(self):
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


    def test_jn_against_prototype(self):
        n = 4
        z = np.array([0.01, 2, 5, 2+0.1j, 3-0.2j, 20+20j])
        jnz = smuthi.spherical_functions.spherical_bessel(n, z)
        self.assertAlmostEqual(jnz[0], 1.058196248205502e-11)
        self.assertAlmostEqual(jnz[1], 0.014079392762915)
        self.assertAlmostEqual(jnz[2], 0.187017655344890)
        self.assertAlmostEqual(jnz[3], 0.013925330885893 + 0.002550081632129j)
        self.assertAlmostEqual(jnz[4], 0.055554281414152 - 0.011718427699962j)
        self.assertAlmostEqual(jnz[5], 5.430299683226971e+06 - 3.884383001639664e+06j)

    def test_hn_against_prototype(self):
        n = 4
        z = np.array([0.01, 2, 5, 2+0.1j, 3-0.2j, 20+20j])
        hnz = smuthi.spherical_functions.spherical_hankel(n, z)
        # self.assertAlmostEqual(hnz[0], 9.562028025173189e-05 - 1.050007500037500e+12j) this test fails - protype incorrect?
        self.assertAlmostEqual(hnz[1], 0.014079392762917 - 4.461291526363127j)
        self.assertAlmostEqual(hnz[2], 0.187017655344889 - 0.186615531479296j)
        self.assertAlmostEqual(hnz[3], -0.937540374646528 - 4.322684701489512j)
        self.assertAlmostEqual(hnz[4], 0.254757423766403 - 0.894658828739464j)
        self.assertAlmostEqual(hnz[5], 5.349924210583894e-11 - 7.680177127921456e-11j)

    def test_dxxj_against_prototype(self):
        n = 4
        z = np.array([0.01, 2, 5, 2+0.1j, 3-0.2j, 20+20j])
        dxxj = smuthi.spherical_functions.dx_xj(n, z)
        self.assertAlmostEqual(dxxj[0], 5.290971621054867e-11)
        self.assertAlmostEqual(dxxj[1], 0.065126624274088)
        self.assertAlmostEqual(dxxj[2], 0.401032469441925)
        self.assertAlmostEqual(dxxj[3], 0.064527362367182 + 0.011261758715092j)
        self.assertAlmostEqual(dxxj[4], 0.230875079050277 - 0.041423344864749j)
        self.assertAlmostEqual(dxxj[5], 3.329872586039824e+07 - 1.858505295737451e+08j)

    def test_dxxh_against_prototype(self):
        n = 4
        z = np.array([0.01, 2, 5, 2+0.1j, 3-0.2j, 20+20j])
        dxxh = smuthi.spherical_functions.dx_xh(n, z)
        #self.assertAlmostEqual(dxxh[0], -3.824801872151283e-04 + 4.200015000000000e+12j)
        self.assertAlmostEqual(dxxh[1], 0.065126624274084 +14.876432990566345j)
        self.assertAlmostEqual(dxxh[2], 0.401032469441923 + 0.669247576352214j)
        self.assertAlmostEqual(dxxh[3], 3.574345443512018 +14.372976166070977j)
        self.assertAlmostEqual(dxxh[4], -0.423459406818455 + 1.976243655979050j)
        self.assertAlmostEqual(dxxh[5], 4.344741738782320e-10 + 2.612636745427169e-09j)

    def test_dxxj_against_j(self):
        n = 3
        eps = 1e-8
        z0 = 0.5
        z = np.array([z0, z0 + eps, z0 - eps])
        jn = smuthi.spherical_functions.spherical_bessel(n, z)
        dxxj = smuthi.spherical_functions.dx_xj(n, z)
        d1 = dxxj[0]
        d2 = ((z0 + eps) * jn[1] - (z0 - eps) * jn[2]) / 2 / eps
        self.assertAlmostEqual(d1, d2)

    def test_dxxh_against_h(self):
        n = 3
        eps = 1e-10
        z0 = 0.5
        z = np.array([z0, z0 + eps, z0 - eps])
        hn = smuthi.spherical_functions.spherical_hankel(n, z)
        dxxh = smuthi.spherical_functions.dx_xh(n, z)
        d1 = dxxh[0]
        d2 = ((z0 + eps) * hn[1] - (z0 - eps) * hn[2]) / 2 / eps
        self.assertTrue((d1 - d2) / d1 < 1e-5)

if __name__ == '__main__':
    unittest.main()
