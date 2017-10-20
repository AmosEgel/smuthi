# -*- coding: utf-8 -*-
import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.particle_coupling as coup

# Parameter input ----------------------------
vacuum_wavelength = 550
beam_polar_angle = np.pi * 7/8
beam_azimuthal_angle = np.pi * 1/3
beam_polarization = 0
beam_amplitude = 1
beam_neff_array = np.linspace(0, 2, 501, endpoint=False)
beam_waist = 1000
beam_focal_point = [200, 200, 200]
neff_waypoints = [0, 0.5, 0.8-0.01j, 2-0.01j, 2.5, 5]
neff_discr = 1e-2
# --------------------------------------------

coord.set_default_k_parallel(vacuum_wavelength, neff_waypoints, neff_discr)

# initialize particle object
sphere1 = part.Sphere(position=[0, 0, 150], refractive_index=2.4 + 0.0j, radius=110, l_max=3, m_max=3)
sphere2 = part.Sphere(position=[305, 0, 150], refractive_index=1.9 + 0.0j, radius=120, l_max=3, m_max=3)
sphere3 = part.Sphere(position=[605, 0, 150], refractive_index=1.7 + 0.0j, radius=90, l_max=3, m_max=3)
particle_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [2, 2, 2])

# lookup = coup.radial_coupling_lookup(vacuum_wavelength, particle_list, lay_sys, resolution=10)
# print(lookup[0][0](105))
#
# wr = coup.layer_mediated_coupling_block(vacuum_wavelength, sphere1, sphere2, lay_sys)
# w = coup.direct_coupling_block(vacuum_wavelength, sphere1, sphere2, lay_sys)
#
# print(w[0, 0] + wr[0, 0])
#
# wr2 = coup.layer_mediated_coupling_block(vacuum_wavelength, sphere1, sphere3, lay_sys)
# w2 = coup.direct_coupling_block(vacuum_wavelength, sphere1, sphere3, lay_sys)
#
# phi13 = np.arctan2(sphere1.position[1] - sphere3.position[1], sphere1.position[0] - sphere3.position[0])
# rho13 = np.sqrt(sum([(sphere1.position[i]-sphere3.position[i])**2 for i in range(3)]))
# print(lookup[0][0](rho13))
# print(w2[0, 0])


# initialize initial field object
init_fld = init.GaussianBeam(vacuum_wavelength=vacuum_wavelength, polar_angle=beam_polar_angle,
                             azimuthal_angle=beam_azimuthal_angle, polarization=beam_polarization,
                             amplitude=beam_amplitude, reference_point=beam_focal_point, beam_waist=beam_waist,
                             k_parallel_array=beam_neff_array*coord.angular_frequency(vacuum_wavelength))

# initialize simulation object
simulation_direct = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld,
                                    solver_type='LU', store_coupling_matrix=True)

simulation_lookup = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld,
                                     solver_type='gmres', store_coupling_matrix=False,
                                     coupling_matrix_lookup_resolution=5)
simulation_lookup.run()
coefficients_lookup = particle_list[0].scattered_field.coefficients
simulation_direct.run()
coefficients_direct = particle_list[0].scattered_field.coefficients

test_vec = np.zeros(90)
test_vec[0] = 1
simulation_lookup.linear_system.coupling_matrix.linear_operator(test_vec)[0]
simulation_direct.linear_system.coupling_matrix.linear_operator(test_vec)[0]

simulation_lookup.linear_system.coupling_matrix.lookup[1][0](0)

def test_result():
    print(coefficients_lookup[0])
    print(coefficients_direct[0])
    relerr = np.linalg.norm(coefficients_lookup - coefficients_direct) / np.linalg.norm(coefficients_direct)
    print('relative error: ', relerr)
    assert relerr < 1e-5
    
if __name__ == '__main__':
    test_result()
