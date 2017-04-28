# -*- coding: utf-8 -*-
"""Test the functions defined in particle_coupling.py."""

import numpy as np
import smuthi.particle_coupling as coup
import smuthi.layers as lay
import smuthi.index_conversion as idx
import smuthi.coordinates as coord
import smuthi.particles as part

idx.set_swe_specs(l_max=2)
wl = 550


def test_wr_against_prototype():
    neff_contour = coord.ComplexContour([0, 0.8, 0.8 - 0.1j, 2.1 - 0.1j, 2.1, 5], 2e-3)

    part_col = part.ParticleCollection()
    part_col.add_sphere(100, 1.7, [100, -100, 200])
    part_col.add_sphere(100, 1.7, [-100, 200, 300])

    laysys_substrate = lay.LayerSystem(thicknesses=[0, 0], refractive_indices=[2 + 0.1j, 1])
    wr_sub = coup.layer_mediated_coupling_matrix(wl, part_col, laysys_substrate, neff_contour)

    wr_sub_0000 = -0.116909038698419 - 0.013001770175717j
    assert abs((wr_sub[0, 0, 0, 0] - wr_sub_0000) / wr_sub_0000) < 1e-5
    wr_sub_0010 = 0.051728301055665 - 0.030410521218822j
    assert abs((wr_sub[0, 0, 1, 0] - wr_sub_0010) / wr_sub_0010) < 1e-5
    wr_sub_0110 = -0.028137473050619 - 0.012620163432327j
    assert abs((wr_sub[0, 1, 1, 0] - wr_sub_0110) / wr_sub_0110) < 1e-5

    laysys_waveguide = lay.LayerSystem(thicknesses=[0, 500, 0], refractive_indices=[1, 2, 1])
    wr_wg = coup.layer_mediated_coupling_matrix(wl, part_col, laysys_waveguide, neff_contour)
    wr_wg_0000 = -0.058321374924359 - 0.030731607595288j
    assert (abs(wr_wg[0, 0, 0, 0] - wr_wg_0000) / wr_wg_0000) < 1e-5
    wr_wg_0010 = -0.065332285111172 - 0.007633555190358j
    assert (abs(wr_wg[0, 0, 1, 0] - wr_wg_0010) / wr_wg_0010) < 1e-5
    wr_wg_0110 = 0.002514697648047 - 0.001765938544514j
    assert (abs(wr_wg[0, 1, 1, 0] - wr_wg_0110) / wr_wg_0110) < 1e-5


def test_w_against_prototype():

    part_col = part.ParticleCollection()
    part_col.add_sphere(100, 1.7, [100, -100, 200])
    part_col.add_sphere(100, 1.7, [-100, 200, 300])
    part_col.add_sphere(100, 1.7, [200, 200, -300])

    laysys_waveguide = lay.LayerSystem(thicknesses=[0, 500, 0], refractive_indices=[1, 2, 1])
    w_wg = coup.direct_coupling_matrix(wl, part_col, laysys_waveguide)
    w_wg_0010 = 0.078085976865533 + 0.054600388160436j
    assert abs((w_wg[0, 0, 1, 0] - w_wg_0010) / w_wg_0010) < 1e-5

    w_wg_0110 = -0.014419231182754 + 0.029269376752105j
    assert abs((w_wg[0, 1, 1, 0] - w_wg_0110) / w_wg_0110) < 1e-5

    w_wg_0912 = -0.118607476554146 + 0.020532217124574j
    assert abs((w_wg[0, 9, 1, 2] - w_wg_0912) / w_wg_0912) < 1e-5

    assert np.linalg.norm(w_wg[0, :, 0, :]) == 0 # no direct self interaction
    assert np.linalg.norm(w_wg[0, :, 2, :]) == 0 # no direct interaction between particles in different layers


def test_w_against_wr():
    neff_contour = coord.ComplexContour([0, 0.8, 0.8 - 0.1j, 2.1 - 0.1j, 2.1, 10], 2e-3)

    laysys_air_1 = lay.LayerSystem(thicknesses=[0, 0], refractive_indices=[1, 1])
    laysys_air_2 = lay.LayerSystem(thicknesses=[0, 250, 0], refractive_indices=[1, 1, 1])

    part_col = part.ParticleCollection()
    part_col.add_sphere(100, 1.7, [100, -100, 200])
    part_col.add_sphere(100, 1.7, [-100, 200, 400])

    w_air_1 = coup.direct_coupling_matrix(wl, part_col, laysys_air_1)
    wr_air_2 = coup.layer_mediated_coupling_matrix(wl, part_col, laysys_air_2, neff_contour)

    np.testing.assert_almost_equal(wr_air_2, w_air_1, decimal=5)


if __name__ == '__main__':
    test_wr_against_prototype()
    test_w_against_prototype()
    test_w_against_wr()
