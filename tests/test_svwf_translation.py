# -*- coding: utf-8 -*-
"""Check wether the translation of spherical wave works properly."""

import numpy as np
import smuthi.linear_system as lin
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.index_conversion as idx
import smuthi.vector_wave_functions as vwf
import smuthi.coordinates as coord


# Parameter input ----------------------------
vacuum_wavelength = 550
surrounding_medium_refractive_index = 1.3
lmax = 10

rx = 100
ry = -100
rz = 100

dx = 400
dy = 800
dz = -500

tau = 0
l = 2
m = -1
# --------------------------------------------

index_specs = idx.swe_specifications(lmax)
k = coord.angular_frequency(vacuum_wavelength) * surrounding_medium_refractive_index

# outgoing wave
Ex, Ey, Ez = vwf.spherical_vector_wave_function(rx + dx, ry + dy, rz + dz, k, 3, tau, l, m)

# series of incoming waves
Ex2, Ey2, Ez2 = complex(0), complex(0), complex(0)
for tau2 in range(2):
    for l2 in range(1, lmax + 1):
        for m2 in range(-l2, l2 + 1):
            Mx, My, Mz = vwf.spherical_vector_wave_function(rx, ry, rz, k, 1, tau2, l2, m2)
            AB = vwf.translation_coefficients_svwf(l, m, l2, m2, k, [dx, dy, dz])
            if tau == tau2:
                Ex2 += Mx * AB[0]
                Ey2 += My * AB[0]
                Ez2 += Mz * AB[0]
            else:
                Ex2 += Mx * AB[1]
                Ey2 += My * AB[1]
                Ez2 += Mz * AB[1]


def test_out_to_reg_expansion():

    assert abs(Ex - Ex2) / abs(Ex) < 1e-4
    assert abs(Ey - Ey2) / abs(Ey) < 1e-4
    assert abs(Ez - Ez2) / abs(Ez) < 1e-4


def test_ab5_versus_prototype():

    a5, b5 = vwf.ab5_coefficients(3, -1, 2, 2, 3)
    a5matl = 0.235702260395516
    assert abs(a5 - a5matl) / abs(a5) < 1e-7

    a5, b5 = vwf.ab5_coefficients(4, 3, 2, 2, 3)
    b5matl = 0.912870929175277j
    assert abs(b5 - b5matl) / abs(b5) < 1e-7

