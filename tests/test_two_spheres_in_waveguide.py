# -*- coding: utf-8 -*-
"""This script runs a simulation for two spheres in a slab waveguide, illuminated by a plane wave."""

import numpy as np
import smuthi.linear_system as lin
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.index_conversion as idx
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.post_processing as pp


# Parameter input ----------------------------
vacuum_wavelength = 550
plane_wave_polar_angle = np.pi
plane_wave_azimuthal_angle = 0
plane_wave_polarization = 0
plane_wave_amplitude = 1
lmax = 3
neff_waypoints = [0, 0.5, 0.8-0.01j, 2-0.01j, 2.5, 20]
neff_discr = 5e-3
farfield_neff_waypoints = [0, 1]
farfield_neff_discr = 1e-2

# --------------------------------------------

idx.set_swe_specs(l_max=lmax)

# initialize particle object
part_col = part.ParticleCollection()
part_col.add_sphere(120, 2.4 + 0.0j, [100, 100, 150])
part_col.add_sphere(120, 1.9 + 0.1j, [-100, -100, 250])

# initialize layer system object
lay_sys1 = lay.LayerSystem([0, 400, 0], [1.5, 1.7, 1])
lay_sys2 = lay.LayerSystem([0, 200, 200, 0], [1.5, 1.7, 1.7, 1])

# initialize initial field object
init_fld = init.InitialFieldCollection(vacuum_wavelength=vacuum_wavelength)
init_fld.add_planewave(amplitude=plane_wave_amplitude, polar_angle=plane_wave_polar_angle,
                       azimuthal_angle=plane_wave_azimuthal_angle, polarization=plane_wave_polarization,
                       reference_point=[0, 0, 400])

# initialize linear system object
lin_sys1 = lin.LinearSystem()
lin_sys2 = lin.LinearSystem()

# initialize simulation object
simulation1 = simul.Simulation(lay_sys1, part_col, init_fld,
                               wr_neff_contour=coord.ComplexContour(neff_waypoints, neff_discr))
simulation1.run()

simulation2 = simul.Simulation(lay_sys2, part_col, init_fld,
                               wr_neff_contour=coord.ComplexContour(neff_waypoints, neff_discr))
simulation2.run()

farfield = pp.scattered_far_field(vacuum_wavelength=vacuum_wavelength,
                                  particle_collection=simulation1.particle_collection,
                                  linear_system=simulation1.linear_system,
                                  layer_system=simulation1.layer_system)


def test_equivalent_layer_systems():
    relerr = (np.linalg.norm(simulation1.linear_system.coupling_matrix[0, :, 1, :] -
                             simulation2.linear_system.coupling_matrix[0, :, 1, :])
              / np.linalg.norm(simulation1.linear_system.coupling_matrix[0, :, 1, :]))
    assert relerr < 1e-3


def test_against_prototype():
    b00 = -0.8609045 + 0.4019615j
    assert abs((simulation1.linear_system.scattered_field_coefficients[0, 0] - b00) / b00) < 1e-4

    b01 = 0.0223519 + 0.0675757j
    assert abs((simulation1.linear_system.scattered_field_coefficients[0, 1] - b01) / b01) < 1e-4

    b027 = -0.0258275 + 0.1145384j
    assert abs((simulation1.linear_system.scattered_field_coefficients[0, 27] - b027) / b027) < 1e-4

    b10 = 0.2065701 + 0.1197903j
    assert abs((simulation1.linear_system.scattered_field_coefficients[1, 0] - b10) / b10) < 1e-4

    top_power_flux = 4.3895865e+02
    assert abs((farfield['forward power'][0] + farfield['forward power'][1] - top_power_flux) / top_power_flux) < 1e-3

    bottom_power_flux = 2.9024410e+04
    assert abs((farfield['backward power'][0] + farfield['backward power'][1] - bottom_power_flux)
               / bottom_power_flux) < 1e-3


if __name__ == '__main__':
    test_equivalent_layer_systems()
    test_against_prototype()
