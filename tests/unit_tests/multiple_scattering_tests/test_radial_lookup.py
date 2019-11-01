# -*- coding: utf-8 -*-
import sys
import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.fields.coordinates_and_contours as coord
import smuthi.simulation as simul
import smuthi.utility.cuda as cu
from smuthi.linearsystem.particlecoupling.direct_coupling import direct_coupling_block
from smuthi.linearsystem.particlecoupling.layer_mediated_coupling import layer_mediated_coupling_block


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
lookup_resol = 5
# --------------------------------------------

coord.set_default_k_parallel(vacuum_wavelength, neff_waypoints, neff_discr)

# initialize particle object
sphere1 = part.Sphere(position=[0, 0, 150], refractive_index=2.4 + 0.1j, radius=100, l_max=3, m_max=3)
sphere2 = part.Sphere(position=[102, -100, 150], refractive_index=1.9 + 0.2j, radius=80, l_max=3, m_max=2)
sphere3 = part.Sphere(position=[202, 100, 150], refractive_index=1.7 + 0.0j, radius=90, l_max=3, m_max=3)
particle_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [1.5, 2 + 0.01j, 1])

phi12 = np.arctan2(sphere1.position[1] - sphere2.position[1], sphere1.position[0] - sphere2.position[0])
rho12 = np.sqrt(sum([(sphere1.position[i]-sphere2.position[i])**2 for i in range(3)]))
wr12 = layer_mediated_coupling_block(vacuum_wavelength, sphere1, sphere2, lay_sys)
w12 = direct_coupling_block(vacuum_wavelength, sphere1, sphere2, lay_sys)

phi13 = np.arctan2(sphere1.position[1] - sphere3.position[1], sphere1.position[0] - sphere3.position[0])
rho13 = np.sqrt(sum([(sphere1.position[i]-sphere3.position[i])**2 for i in range(3)])) 
wr13 = layer_mediated_coupling_block(vacuum_wavelength, sphere1, sphere3, lay_sys)
w13 = direct_coupling_block(vacuum_wavelength, sphere1, sphere3, lay_sys)
 
# initialize initial field object
init_fld = init.GaussianBeam(vacuum_wavelength=vacuum_wavelength, polar_angle=beam_polar_angle,
                             azimuthal_angle=beam_azimuthal_angle, polarization=beam_polarization,
                             amplitude=beam_amplitude, reference_point=beam_focal_point, beam_waist=beam_waist,
                             k_parallel_array=beam_neff_array*coord.angular_frequency(vacuum_wavelength))

# initialize simulation object
simulation_direct = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld,
                                     solver_type='LU', store_coupling_matrix=True,
                                     log_to_terminal=(not sys.argv[0].endswith('nose2')))  # suppress output if called by nose
simulation_direct.run()
coefficients_direct = particle_list[0].scattered_field.coefficients

cu.enable_gpu()
simulation_lookup_linear_gpu = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, 
                                                initial_field=init_fld, solver_type='gmres', store_coupling_matrix=False,
                                                coupling_matrix_lookup_resolution=lookup_resol, 
                                                coupling_matrix_interpolator_kind='linear',
                                                log_to_terminal=(not sys.argv[0].endswith('nose2')))  # suppress output if called by nose
simulation_lookup_linear_gpu.run()
coefficients_lookup_linear_gpu = particle_list[0].scattered_field.coefficients

simulation_lookup_cubic_gpu = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, 
                                               initial_field=init_fld, solver_type='gmres', store_coupling_matrix=False,
                                               coupling_matrix_lookup_resolution=lookup_resol, 
                                               coupling_matrix_interpolator_kind='cubic',
                                               log_to_terminal=(not sys.argv[0].endswith('nose2')))  # suppress output if called by nose
simulation_lookup_cubic_gpu.run()
coefficients_lookup_cubic_gpu = particle_list[0].scattered_field.coefficients

cu.enable_gpu(False)
simulation_lookup_linear_cpu = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, 
                                                initial_field=init_fld, solver_type='gmres', store_coupling_matrix=False,
                                                coupling_matrix_lookup_resolution=lookup_resol, 
                                                coupling_matrix_interpolator_kind='linear',
                                                log_to_terminal=(not sys.argv[0].endswith('nose2')))  # suppress output if called by nose
simulation_lookup_linear_cpu.run()
coefficients_lookup_linear_cpu = particle_list[0].scattered_field.coefficients

simulation_lookup_cubic_cpu = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, 
                                                initial_field=init_fld, solver_type='gmres', store_coupling_matrix=False,
                                                coupling_matrix_lookup_resolution=lookup_resol, 
                                                coupling_matrix_interpolator_kind='cubic',
                                                log_to_terminal=(not sys.argv[0].endswith('nose2')))  # suppress output if called by nose
simulation_lookup_cubic_cpu.run()
coefficients_lookup_cubic_cpu = particle_list[0].scattered_field.coefficients

test_vec = np.arange(simulation_lookup_linear_cpu.linear_system.master_matrix.shape[0])
M_direct_test_vec = simulation_direct.linear_system.coupling_matrix.linear_operator(test_vec)
M_linear_cpu_test_vec = simulation_lookup_linear_cpu.linear_system.coupling_matrix.linear_operator(test_vec)
M_cubic_cpu_test_vec = simulation_lookup_cubic_cpu.linear_system.coupling_matrix.linear_operator(test_vec)
M_linear_gpu_test_vec = simulation_lookup_linear_gpu.linear_system.coupling_matrix.linear_operator(test_vec)
M_cubic_gpu_test_vec = simulation_lookup_cubic_gpu.linear_system.coupling_matrix.linear_operator(test_vec)


def test_linear_operator():
    relerr = np.linalg.norm(M_linear_cpu_test_vec - M_direct_test_vec) / np.linalg.norm(M_direct_test_vec)
    print('relative error linear operator linear interpoloation CPU: ', relerr)
    assert relerr < 1e-2
    
    relerr = np.linalg.norm(M_cubic_cpu_test_vec - M_direct_test_vec) / np.linalg.norm(M_direct_test_vec)
    print('relative error linear operator cubic interpoloation CPU: ', relerr)
    assert relerr < 5e-4
    
    relerr = np.linalg.norm(M_linear_gpu_test_vec - M_direct_test_vec) / np.linalg.norm(M_direct_test_vec)
    print('relative error linear operator linear interpoloation GPU: ', relerr)
    assert relerr < 1e-2
    
    relerr = np.linalg.norm(M_cubic_gpu_test_vec - M_direct_test_vec) / np.linalg.norm(M_direct_test_vec)
    print('relative error linear operator cubic interpoloation GPU: ', relerr)
    assert relerr < 5e-4    


def test_result():
    relerr = np.linalg.norm(coefficients_lookup_linear_cpu - coefficients_direct) / np.linalg.norm(coefficients_direct)
    print('relative error coefficient solution linear interpolation CPU: ', relerr)
    assert relerr < 5e-4
    
    relerr = np.linalg.norm(coefficients_lookup_cubic_cpu - coefficients_direct) / np.linalg.norm(coefficients_direct)
    print('relative error coefficient solution cubic interpolation CPU: ', relerr)
    assert relerr < 5e-4
    
    relerr = np.linalg.norm(coefficients_lookup_linear_gpu - coefficients_direct) / np.linalg.norm(coefficients_direct)
    print('relative error coefficient solution linear interpolation GPU: ', relerr)
    assert relerr < 5e-4
    
    relerr = np.linalg.norm(coefficients_lookup_cubic_gpu - coefficients_direct) / np.linalg.norm(coefficients_direct)
    print('relative error coefficient solution cubic interpolation GPU: ', relerr)
    assert relerr < 5e-4    


if __name__ == '__main__':
    test_linear_operator()
    test_result()
