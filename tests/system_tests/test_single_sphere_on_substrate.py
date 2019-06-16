# -*- coding: utf-8 -*-
"""This script runs a simulation for a single sphere on a substrate, illuminated by a plane wave."""

import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.simulation as sim
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

coord.set_default_k_parallel(vacuum_wavelength, neff_waypoints, neff_discr)

# initialize particle object
part1 = part.Sphere(position=[0, 0, distance_sphere_substrate + sphere_radius],
                    refractive_index=sphere_refractive_index, radius=sphere_radius, l_max=lmax, m_max=lmax)
particle_list = [part1]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 0], [substrate_refractive_index, surrounding_medium_refractive_index])

# initialize initial field object
init_fld = init.PlaneWave(vacuum_wavelength=vacuum_wavelength, polar_angle=plane_wave_polar_angle,
                          azimuthal_angle=plane_wave_azimuthal_angle, polarization=plane_wave_polarization,
                          amplitude=plane_wave_amplitude, reference_point=[0, 0, 0])

# simulation
simulation = sim.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld, log_to_terminal=False)

simulation.run()


def test_versus_prototype():
    b0 = -0.2586209 + 0.8111274j
    assert abs((particle_list[0].scattered_field.coefficients[0] - b0) / b0) < 1e-4
    b10 = -1.5103858e-04 - 4.1782795e-04j
    assert abs((particle_list[0].scattered_field.coefficients[10] - b10) / b10) < 1e-4
    b21 = -0.0795316 + 0.0194518j
    assert abs((particle_list[0].scattered_field.coefficients[21] - b21) / b21) < 1e-4


if __name__ == '__main__':
    test_versus_prototype()
