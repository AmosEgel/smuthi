# -*- coding: utf-8 -*-

import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.scattered_field as sf


# Parameter input ----------------------------
vacuum_wavelength = 550
plane_wave_polar_angle = np.pi * 7/8
plane_wave_azimuthal_angle = np.pi * 1/3
plane_wave_polarization = 0
plane_wave_amplitude = 1
neff_waypoints = [0, 0.5, 0.8-0.01j, 2-0.01j, 2.5, 5]
neff_discr = 5e-3
# --------------------------------------------

coord.set_default_k_parallel(vacuum_wavelength, neff_waypoints, neff_discr)

# initialize particle object
sphere1 = part.Sphere(position=[100, 100, 150], refractive_index=2.4 + 0.0j, radius=110, l_max=4, m_max=4)
sphere2 = part.Sphere(position=[-100, -100, 250], refractive_index=1.9 + 0.0j, radius=120, l_max=3, m_max=3)
sphere3 = part.Sphere(position=[-200, 100, 300], refractive_index=1.7 + 0.0j, radius=90, l_max=3, m_max=3)
particle_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [2, 1.3, 2])

# initialize initial field object
init_fld = init.PlaneWave(vacuum_wavelength=vacuum_wavelength, polar_angle=plane_wave_polar_angle,
                          azimuthal_angle=plane_wave_azimuthal_angle, polarization=plane_wave_polarization,
                          amplitude=plane_wave_amplitude, reference_point=[0, 0, 400])

# initialize simulation object
simulation = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld, log_to_terminal=False)
simulation.run()

scs = sf.scattering_cross_section(initial_field=simulation.initial_field, particle_list=simulation.particle_list,
                                  layer_system=simulation.layer_system)

ecs = sf.extinction_cross_section(initial_field=simulation.initial_field,particle_list=simulation.particle_list,
                                  layer_system=simulation.layer_system)


def test_optical_theorem():
    relerr = abs((sum(scs.integral()) - ecs['top'] - ecs['bottom']) / sum(scs.integral()))
    print('error: ', relerr)
    assert relerr < 1e-4


if __name__ == '__main__':
    test_optical_theorem()
