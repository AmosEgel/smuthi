# -*- coding: utf-8 -*-
"""This script runs a simulation for a single sphere on a substrate, illuminated by a plane wave."""

import numpy as np
import smuthi.linear_system as lin
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.index_conversion as idx
import smuthi.particle_coupling as coup


# Parameter input ----------------------------
vacuum_wavelength = 550
sphere_radius = 100
surrounding_medium_refractive_index = 1
substrate_refractive_index = 1.52
plane_wave_polar_angle = np.pi
plane_wave_azimuthal_angle = 0
plane_wave_polarization = 0
plane_wave_amplitude = 1
lmax = 3

# --------------------------------------------

idx.l_max = lmax
idx.m_max = lmax

# initialize particle object
part_col = part.ParticleCollection()
part_col.add_sphere(100, 1.7, [100, -100, 200])
part_col.add_sphere(100, 1.7, [-100, 200, 300])
part_col.add_sphere(100, 1.7, [-100, -200, -300])

# initialize layer system object
lay_sys = lay.LayerSystem([0, 0], [substrate_refractive_index, surrounding_medium_refractive_index])

# initialize equation object
lin_sys = lin.LinearSystem()
coupling_matrix = coup.direct_coupling_matrix(vacuum_wavelength, part_col, lay_sys)


def test_versus_prototype():
    w12_00 = -0.163409214869119 + 0.112130245600907j
    assert abs((coupling_matrix[0, 0, 1, 0] - w12_00) / w12_00) < 1e-4
    w12_2_16 = 0.175494924814760 + 0.171772249467886j
    assert abs((coupling_matrix[0, 2, 1, 16] - w12_2_16) / w12_2_16) < 1e-4
    assert abs(coupling_matrix[0, 0, 2, 0]) == 0  # zero direct coupling to particle in different layer


if __name__ == '__main__':
    test_versus_prototype()
