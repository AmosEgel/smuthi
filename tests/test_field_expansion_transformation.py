# -*- coding: utf-8 -*-

import numpy as np
import smuthi.field_expansion as fldex
import smuthi.layers as lay

omega = 2 * 3.15 / 550
refractive_index = 1
k = omega * refractive_index

kp_vec1 = np.linspace(0, 0.99, 500) * k
kp_vec2 = 1j * np.linspace(0, -0.01, 100) * k + 0.99 * k
kp_vec3 = np.linspace(0, 0.02, 200) * k + (0.99 - 0.01j) * k
kp_vec4 = 1j * np.linspace(0, 0.01, 100) * k + (1.01 - 0.01j) * k
kp_vec5 = np.linspace(0, 6, 1000) * k + 1.01 * k
kp = np.concatenate([kp_vec1, kp_vec2, kp_vec3, kp_vec4, kp_vec5])

a = np.linspace(0, 2 * np.pi, num=200)
pwe_ref = [-100, -100, -100]
swe_ref = [-400, 200, 500]
fieldpoint = [-500, 310, 620]
vb = [0, 800]

layer_system = lay.LayerSystem([0, 0], [1, 1])

x = np.array([fieldpoint[0]])
y = np.array([fieldpoint[1]])
z = np.array([fieldpoint[2]])

swe = fldex.SphericalWaveExpansion(k=k, l_max=3, m_max=3, type='outgoing', reference_point=swe_ref)
swe.coefficients[0] = 2
swe.coefficients[1] = -3j
swe.coefficients[16] = 1
swe.coefficients[18] = 0.5

kp2 = np.linspace(0, 2, num=1000) * k
pwe = fldex.PlaneWaveExpansion(k=k, k_parallel=kp2, azimuthal_angles=a, type='upgoing', reference_point=pwe_ref,
                               valid_between=vb)
pwe.coefficients[0, :, :] = 100000 * np.exp(- pwe.k_parallel_grid() / k / 20)
pwe.coefficients[1, :, :] = -100000 * np.exp(- pwe.k_parallel_grid() / k / 10)


def test_swe2pwe():
    ex, ey, ez = swe.electric_field(x, y, z)
    pwe_up, pwe_down = fldex.swe_to_pwe_conversion(swe, k_parallel=kp, azimuthal_angles=a, layer_system=layer_system)
    ex2, ey2, ez2 = pwe_up.electric_field(x, y, z)
    err2 = abs(ex - ex2) ** 2 + abs(ey - ey2) ** 2 + abs(ez - ez2) ** 2
    norme2 = abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2
    # print(ex, ey, ez)
    # print(ex2, ey2, ez2)
    # print(np.sqrt(err2 / norme2))
    assert np.sqrt(err2 / norme2) < 5e-3


def test_pwe2swe():
    ex4, ey4, ez4 = pwe.electric_field(x, y, z)
    swe_reg = fldex.pwe_to_swe_conversion(pwe, 6, 6, reference_point=swe_ref)
    ex5, ey5, ez5 = swe_reg.electric_field(x, y, z)
    err2 = abs(ex4 - ex5)**2 + abs(ey4 - ey5)**2 + abs(ez4 - ez5)**2
    norme2 = abs(ex4)**2 + abs(ey4)**2 + abs(ez4)**2
    # print(ex4,  ey4, ez4)
    # print(ex5,  ey5, ez5)
    # print(np.sqrt(err2/norme2))
    assert np.sqrt(err2 / norme2) < 5e-3


if __name__ == '__main__':
    test_swe2pwe()
    test_pwe2swe()
