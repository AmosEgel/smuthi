# -*- coding: utf-8 -*-

import sys
import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.simulation as simul
import smuthi.postprocessing.far_field as farf
import smuthi.utility.automatic_parameter_selection as autoparam
import smuthi.fields as flds


# Parameter input ----------------------------
vacuum_wavelength = 550
plane_wave_polar_angle = np.pi * 7/8
plane_wave_azimuthal_angle = np.pi * 1/3
plane_wave_polarization = 0
plane_wave_amplitude = 1
# --------------------------------------------

# initialize particle objects
sphere1 = part.Sphere(position=[100, 100, 150], refractive_index=2.4 + 0.0j, radius=110, l_max=1, m_max=1)
sphere2 = part.Sphere(position=[-100, -100, 250], refractive_index=1.9 + 0.0j, radius=120, l_max=1, m_max=1)
sphere3 = part.Sphere(position=[-200, 100, 300], refractive_index=1.7 + 0.0j, radius=90, l_max=1, m_max=1)
particle_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [2, 1.3, 2])

# initialize initial field object
init_fld = init.PlaneWave(vacuum_wavelength=vacuum_wavelength, polar_angle=plane_wave_polar_angle,
                          azimuthal_angle=plane_wave_azimuthal_angle, polarization=plane_wave_polarization,
                          amplitude=plane_wave_amplitude, reference_point=[0, 0, 400])

# initialize simulation object
simulation = simul.Simulation(layer_system=lay_sys,
                              particle_list=particle_list,
                              initial_field=init_fld,
                              log_to_terminal=(not sys.argv[0].endswith('nose2')))  # suppress output if called by nose

autoparam.select_numerical_parameters(simulation,
                                      detector="extinction cross section",
                                      tolerance=1e-5,
                                      max_iter=20,
                                      neff_imag=1e-2,
                                      neff_step=1e-2,
                                      select_neff_max=True,
                                      neff_max_increment=0.5,
                                      neff_max=None,
                                      select_neff_step=True,
                                      select_multipole_cutoff=True,
                                      relative_convergence=True,
                                      suppress_simulation_output=True)

simulation.run()

scs = farf.scattering_cross_section(initial_field=simulation.initial_field, particle_list=simulation.particle_list,
                                  layer_system=simulation.layer_system)

ecs = farf.extinction_cross_section(initial_field=simulation.initial_field,particle_list=simulation.particle_list,
                                  layer_system=simulation.layer_system)


def test_optical_theorem():
    relerr = abs((sum(scs.integral()) - ecs['top'] - ecs['bottom']) / sum(scs.integral()))
    print('error: ', relerr)
    assert relerr < 1e-4


if __name__ == '__main__':
    test_optical_theorem()
