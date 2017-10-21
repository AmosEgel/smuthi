# -*- coding: utf-8 -*-
import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.particle_coupling as coup
import smuthi.field_expansion as fldex

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
sphere1 = part.Sphere(position=[0, 0, 150], refractive_index=2.4 + 0.0j, radius=40, l_max=3, m_max=3)
sphere2 = part.Sphere(position=[102, 0, 150], refractive_index=1.9 + 0.0j, radius=40, l_max=3, m_max=2)
sphere3 = part.Sphere(position=[202, 100, 150], refractive_index=1.7 + 0.0j, radius=40, l_max=3, m_max=3)
particle_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [1.5, 2 + 0.01j, 1])

lookup = coup.radial_coupling_lookup(vacuum_wavelength, particle_list, lay_sys, resolution=lookup_resol)

phi12 = np.arctan2(sphere1.position[1] - sphere2.position[1], sphere1.position[0] - sphere2.position[0])
rho12 = np.sqrt(sum([(sphere1.position[i]-sphere2.position[i])**2 for i in range(3)]))
wr12 = coup.layer_mediated_coupling_block(vacuum_wavelength, sphere1, sphere2, lay_sys)
w12 = coup.direct_coupling_block(vacuum_wavelength, sphere1, sphere2, lay_sys)

phi13 = np.arctan2(sphere1.position[1] - sphere3.position[1], sphere1.position[0] - sphere3.position[0])
rho13 = np.sqrt(sum([(sphere1.position[i]-sphere3.position[i])**2 for i in range(3)])) 
wr13 = coup.layer_mediated_coupling_block(vacuum_wavelength, sphere1, sphere3, lay_sys)
w13 = coup.direct_coupling_block(vacuum_wavelength, sphere1, sphere3, lay_sys)
 
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
                                     coupling_matrix_lookup_resolution=lookup_resol)
simulation_lookup.run()
coefficients_lookup = particle_list[0].scattered_field.coefficients
simulation_direct.run()
coefficients_direct = particle_list[0].scattered_field.coefficients

test_vec = np.arange(simulation_lookup.linear_system.master_matrix.shape[0])
test_vec[0] = 1
M_lookup_test_vec = simulation_lookup.linear_system.coupling_matrix.linear_operator(test_vec)
M_direct_test_vec = simulation_direct.linear_system.coupling_matrix.linear_operator(test_vec)


def test_lookup():
    
    tau1 = 0
    l1 = 3
    m1 = -1
    n1_look = fldex.multi_to_single_index(tau1, l1, m1, simulation_lookup.linear_system.coupling_matrix.l_max, 
                                          simulation_lookup.linear_system.coupling_matrix.m_max)
    n1 = fldex.multi_to_single_index(tau1, l1, m1, sphere1.l_max, sphere1.m_max)
    
    tau2 =1
    l2 = 2
    m2 = 2
    n2_look = fldex.multi_to_single_index(tau2, l2, m2, simulation_lookup.linear_system.coupling_matrix.l_max, 
                                          simulation_lookup.linear_system.coupling_matrix.m_max)
    n2 = fldex.multi_to_single_index(tau2, l2, m2, sphere2.l_max, sphere2.m_max)
    
    tau3 =0
    l3 = 3
    m3 = 3
    n3_look = fldex.multi_to_single_index(tau3, l3, m3, simulation_lookup.linear_system.coupling_matrix.l_max, 
                                          simulation_lookup.linear_system.coupling_matrix.m_max)
    n3 = fldex.multi_to_single_index(tau3, l3, m3, sphere3.l_max, sphere3.m_max)
    
    wl12 = lookup[n1_look][n2_look](rho12) * np.exp(1j * (m2 - m1) * phi12)
    wd12 = w12[n1, n2] + wr12[n1, n2]
    relerr12 = abs(wl12 - wd12) / abs(wl12)
    print('relative error w12 ', relerr12)
    assert relerr12 < 5e-4
        
    wl13 = lookup[n1_look][n3_look](rho13) * np.exp(1j * (m3 - m1) * phi13)
    wd13 = w13[n1, n3] + wr13[n1, n3]
    relerr13 = abs(wl13 - wd13) / abs(wl13)
    print('relative error w13 ', relerr13)
    assert relerr13 < 5e-4
    

def test_linear_operator():
    relerr = np.linalg.norm(M_lookup_test_vec - M_direct_test_vec) / np.linalg.norm(M_direct_test_vec)
    print('relative error linear operator: ', relerr)
    assert relerr < 5e-4
            
def test_result():
    relerr = np.linalg.norm(coefficients_lookup - coefficients_direct) / np.linalg.norm(coefficients_direct)
    print('relative error coefficient solution: ', relerr)
    assert relerr < 5e-4
    
if __name__ == '__main__':
    test_lookup()
    test_linear_operator()
    test_result()
