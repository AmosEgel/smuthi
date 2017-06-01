# -*- coding: utf-8 -*-
"""This script runs a simulation for a single sphere on a substrate, illuminated by a plane wave."""

import numpy as np
import smuthi.linear_system as lin
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.index_conversion as idx
import smuthi.t_matrix as tmt
import smuthi.particle_coupling as coup
import smuthi.coordinates as coord


# Parameter input ----------------------------
vacuum_wavelength = 550
sphere_radius = 100
surrounding_medium_refractive_index = 1
substrate_refractive_index = 1.52
sphere_refractive_index = 2.4
distance_sphere_substrate = 50
plane_wave_polar_angle = np.pi
plane_wave_azimuthal_angle = 0
plane_wave_polarization = 0
plane_wave_amplitude = 1
lmax = 3
neff_waypoints = [0, 0.5, 0.8-0.1j, 2-0.1j, 2.5, 4]
neff_discr = 1e-3
farfield_neff_waypoints = [0, 1]
farfield_neff_discr = 1e-2

# --------------------------------------------

idx.set_swe_specs(l_max=lmax)

# initialize particle object
part_col = part.ParticleCollection()
part_col.add_sphere(sphere_radius, sphere_refractive_index, [0, 0, distance_sphere_substrate + sphere_radius])

# initialize layer system object
lay_sys = lay.LayerSystem([0, 0], [substrate_refractive_index, surrounding_medium_refractive_index])

# initialize initial field object
init_fld = init.InitialFieldCollection(vacuum_wavelength=vacuum_wavelength)
init_fld.add_planewave(amplitude=plane_wave_amplitude, polar_angle=plane_wave_polar_angle,
                       azimuthal_angle=plane_wave_azimuthal_angle, polarization=plane_wave_polarization)

# initialize equation object
lin_sys = lin.LinearSystem()

# compute initial field coefficients
lin_sys.initial_field_coefficients = init.initial_field_swe_coefficients(init_fld, part_col, lay_sys)

# compute T-matrix
lin_sys.t_matrices = tmt.t_matrix_collection(vacuum_wavelength, part_col, lay_sys)

# compute particle coupling matrix
neff_contour = coord.ComplexContour(neff_waypoints, neff_discr)
lin_sys.coupling_matrix = coup.layer_mediated_coupling_matrix(vacuum_wavelength, part_col, lay_sys, neff_contour)

# solve linear system
lin_sys.solve()


def test_versus_prototype():
    b0 = -0.2586209 + 0.8111274j
    assert abs((lin_sys.scattered_field_coefficients[0, 0] - b0) / b0) < 1e-4
    b10 = -1.5103858e-04 - 4.1782795e-04j
    assert abs((lin_sys.scattered_field_coefficients[0, 10] - b10) / b10) < 1e-4
    b21 = -0.0795316 + 0.0194518j
    assert abs((lin_sys.scattered_field_coefficients[0, 21] - b21) / b21) < 1e-4


if __name__ == '__main__':
    test_versus_prototype()
