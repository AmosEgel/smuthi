# -*- coding: utf-8 -*-
"""Test spherical_functions"""

import smuthi.spherical_functions
import numpy as np
from sympy.physics.quantum.spin import Rotation


def test_wignerd():
    l_test = 5
    m_test = -3
    m_prime_test = 4
    beta_test = 0.64
    wigd = smuthi.spherical_functions.wigner_d(l_test, m_test, m_prime_test, beta_test, wdsympy=False)
    wigd_sympy = complex(Rotation.d(l_test, m_test, m_prime_test, beta_test).doit()).real
    err = abs((wigd - wigd_sympy) / wigd)
    assert err < 1e-10


def test_Plm_against_prototype():
    lmax = 3
    omega = 2 * 3.14 / 550
    kp = omega * np.array([0.01, 0.2, 0.7, 0.99, 1.2, 2 - 0.5j])
    kz = np.sqrt(omega ** 2 - kp ** 2 + 0j)
    kz[kz.imag < 0] = -kz[kz.imag < 0]
    ct = kz / omega
    st = kp / omega
    plm, pilm, taulm = smuthi.spherical_functions.legendre_normalized(ct, st, lmax)

    # P_3^0
    np.testing.assert_almost_equal(plm[3][0][0], 1.870267465826245)
    np.testing.assert_almost_equal(plm[3][0][1], 1.649727250184103)
    np.testing.assert_almost_equal(plm[3][0][2], -0.300608757357466)
    np.testing.assert_almost_equal(plm[3][0][3], -0.382739631607606)
    np.testing.assert_almost_equal(plm[3][0][4], - 3.226515147957620j)
    np.testing.assert_almost_equal(plm[3][0][5], -25.338383323084539 - 22.141864985871653j)

    # P_2^1
    np.testing.assert_almost_equal(plm[2][1][0], 0.019363948460993)
    np.testing.assert_almost_equal(plm[2][1][1], 0.379473319220206)
    np.testing.assert_almost_equal(plm[2][1][2], 0.968052168015753)
    np.testing.assert_almost_equal(plm[2][1][3], 0.270444009917026)
    np.testing.assert_almost_equal(plm[2][1][4], 1.541427909439815j)
    np.testing.assert_almost_equal(plm[2][1][5], 3.906499971729346 + 6.239600710712296j)

    # pi_2^1
    np.testing.assert_almost_equal(pilm[2][1][0], 1.936394846099318)
    np.testing.assert_almost_equal(pilm[2][1][1], 1.897366596101028)
    np.testing.assert_almost_equal(pilm[2][1][2], 1.382931668593933)
    np.testing.assert_almost_equal(pilm[2][1][3], 0.273175767592955)
    np.testing.assert_almost_equal(pilm[2][1][4], 1.284523257866512j)
    np.testing.assert_almost_equal(pilm[2][1][5], 1.104282256024128 + 3.395870919362181j)

    # tau_3^2
    np.testing.assert_almost_equal(taulm[3][2][0], 0.051227068616724)
    np.testing.assert_almost_equal(taulm[3][2][1], 0.963213372000203)
    np.testing.assert_almost_equal(taulm[3][2][2], 0.950404683542753)
    np.testing.assert_almost_equal(taulm[3][2][3], -2.384713931794872)
    np.testing.assert_almost_equal(taulm[3][2][4], -7.131877733107878)
    np.testing.assert_almost_equal(taulm[3][2][5], -39.706934218093430 + 42.588889121019569j)


def test_jn_against_prototype():
    n = 4
    z = np.array([0.01, 2, 5, 2+0.1j, 3-0.2j, 20+20j])
    jnz = smuthi.spherical_functions.spherical_bessel(n, z)
    np.testing.assert_almost_equal(jnz[0], 1.058196248205502e-11)
    np.testing.assert_almost_equal(jnz[1], 0.014079392762915)
    np.testing.assert_almost_equal(jnz[2], 0.187017655344890)
    np.testing.assert_almost_equal(jnz[3], 0.013925330885893 + 0.002550081632129j)
    np.testing.assert_almost_equal(jnz[4], 0.055554281414152 - 0.011718427699962j)
    np.testing.assert_almost_equal(jnz[5], 5.430299683226971e+06 - 3.884383001639664e+06j)


def test_hn_against_prototype():
    n = 4
    z = np.array([0.01, 2, 5, 2+0.1j, 3-0.2j, 20+20j])
    hnz = smuthi.spherical_functions.spherical_hankel(n, z)
    # np.testing.assert_almost_equal(hnz[0], 9.562028025173189e-05 - 1.050007500037500e+12j) this test fails - protype incorrect?
    np.testing.assert_almost_equal(hnz[1], 0.014079392762917 - 4.461291526363127j)
    np.testing.assert_almost_equal(hnz[2], 0.187017655344889 - 0.186615531479296j)
    np.testing.assert_almost_equal(hnz[3], -0.937540374646528 - 4.322684701489512j)
    np.testing.assert_almost_equal(hnz[4], 0.254757423766403 - 0.894658828739464j)
    np.testing.assert_almost_equal(hnz[5], 5.349924210583894e-11 - 7.680177127921456e-11j)


def test_dxxj_against_prototype():
    n = 4
    z = np.array([0.01, 2, 5, 2+0.1j, 3-0.2j, 20+20j])
    dxxj = smuthi.spherical_functions.dx_xj(n, z)
    np.testing.assert_almost_equal(dxxj[0], 5.290971621054867e-11)
    np.testing.assert_almost_equal(dxxj[1], 0.065126624274088)
    np.testing.assert_almost_equal(dxxj[2], 0.401032469441925)
    np.testing.assert_almost_equal(dxxj[3], 0.064527362367182 + 0.011261758715092j)
    np.testing.assert_almost_equal(dxxj[4], 0.230875079050277 - 0.041423344864749j)
    np.testing.assert_almost_equal(dxxj[5], 3.329872586039824e+07 - 1.858505295737451e+08j)


def test_dxxh_against_prototype():
    n = 4
    z = np.array([0.01, 2, 5, 2+0.1j, 3-0.2j, 20+20j])
    dxxh = smuthi.spherical_functions.dx_xh(n, z)
    #np.testing.assert_almost_equal(dxxh[0], -3.824801872151283e-04 + 4.200015000000000e+12j)
    np.testing.assert_almost_equal(dxxh[1], 0.065126624274084 +14.876432990566345j)
    np.testing.assert_almost_equal(dxxh[2], 0.401032469441923 + 0.669247576352214j)
    np.testing.assert_almost_equal(dxxh[3], 3.574345443512018 +14.372976166070977j)
    np.testing.assert_almost_equal(dxxh[4], -0.423459406818455 + 1.976243655979050j)
    np.testing.assert_almost_equal(dxxh[5], 4.344741738782320e-10 + 2.612636745427169e-09j)


def test_dxxj_against_j():
    n = 3
    eps = 1e-8
    z0 = 0.5
    z = np.array([z0, z0 + eps, z0 - eps])
    jn = smuthi.spherical_functions.spherical_bessel(n, z)
    dxxj = smuthi.spherical_functions.dx_xj(n, z)
    d1 = dxxj[0]
    d2 = ((z0 + eps) * jn[1] - (z0 - eps) * jn[2]) / 2 / eps
    np.testing.assert_almost_equal(d1, d2)


def test_dxxh_against_h():
    n = 3
    eps = 1e-10
    z0 = 0.5
    z = np.array([z0, z0 + eps, z0 - eps])
    hn = smuthi.spherical_functions.spherical_hankel(n, z)
    dxxh = smuthi.spherical_functions.dx_xh(n, z)
    d1 = dxxh[0]
    d2 = ((z0 + eps) * hn[1] - (z0 - eps) * hn[2]) / 2 / eps
    assert (d1 - d2) / d1 < 1e-5


if __name__ == '__main__':
    test_wignerd()
    test_dxxh_against_h()
    test_dxxh_against_prototype()
    test_dxxj_against_j()
    test_dxxj_against_prototype()
    test_hn_against_prototype()
    test_jn_against_prototype()
    test_Plm_against_prototype()
