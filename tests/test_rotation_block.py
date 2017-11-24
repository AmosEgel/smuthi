# -*- coding: utf-8 -*-
import numpy as np
import smuthi.vector_wave_functions as vwf
import smuthi.spherical_functions as sf
import smuthi.particles as part
import smuthi.simulation as simul
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.coordinates as coord
import smuthi.scattered_field as scf


# Parameter input ----------------------------
vacuum_wavelength = 550
surrounding_medium_refractive_index = 1.3

r = np.array([-800, -200, 100])

alpha = np.pi * 0.1
beta = np.pi * 0.7
gamma = np.pi * 0.7 

tau = 1
l = 11
m = -7
k = 0.0126
                           
r_prime = coord.vector_rotation(r, euler_angles=[alpha, beta, gamma])

# outgoing wave (rotated coordinate system)
E_prime = vwf.spherical_vector_wave_function(r_prime[0], r_prime[1], r_prime[2], k, 3, tau, l, m)

# transformation into the laboratory coordinate system
E = coord.inverse_vector_rotation(E_prime, euler_angles=[alpha, beta, gamma])

# 
Ex2, Ey2, Ez2 = complex(0), complex(0), complex(0)
Mx = np.zeros([2 * l + 1], dtype=complex)
My = np.zeros([2 * l + 1], dtype=complex)
Mz = np.zeros([2 * l + 1], dtype=complex)
D = np.zeros([2 * l + 1], dtype=complex)

cc = 0
for m_prime in range(-l, l + 1):
    Mx[cc], My[cc], Mz[cc] = vwf.spherical_vector_wave_function(r[0], r[1], r[2], k, 3, tau, l, m_prime)
    D[cc] = sf.wigner_D(l, m, m_prime, -gamma, -beta, -alpha, False)
    Ex2 += Mx[cc] * D[cc] 
    Ey2 += My[cc] * D[cc]
    Ez2 += Mz[cc] * D[cc]
    cc += 1


def test_rotation():
    err = abs((E - np.array([Ex2, Ey2, Ez2])) / E)
    print('error Rotation-Addition-Theorem', err)
    assert err[0] < 1e-4
    assert err[1] < 1e-4
    assert err[2] < 1e-4

    
ld = 550
rD = [0, 0, 200]
rD2 = [0, 0, 200]
waypoints = [0, 0.8, 0.8-0.1j, 2.1-0.1j, 2.1, 4]
neff_discr = 2e-2

coord.set_default_k_parallel(vacuum_wavelength = ld, neff_waypoints=waypoints, neff_resolution=neff_discr)


# initialize particle objects
spheroid1 = part.Spheroid(position=[0, 0, 200], euler_angles=[0, 0, 0], refractive_index=2.4, semi_axis_c=100,
                          semi_axis_a=50, l_max=8, m_max=8, t_matrix_method=None)
spheroid2 = part.Spheroid(position=[0, 0, 200], euler_angles=[0.425 * np.pi, 0.25 * np.pi, 0], refractive_index=2.4,
                          semi_axis_c=100, semi_axis_a=50, l_max=8, m_max=8, t_matrix_method=None)
part_list = [spheroid1]
part_list2 = [spheroid2]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 800, 0], [1, 1, 1])

# initialize plane wave objects
planewave = init.PlaneWave(vacuum_wavelength=ld, polar_angle=np.pi, azimuthal_angle=0, polarization=0, amplitude=1,
                           reference_point=rD)
planewave2 = init.PlaneWave(vacuum_wavelength=ld, polar_angle=0.75 * np.pi, azimuthal_angle=1.425 *np.pi, polarization=0,
                            amplitude=1, reference_point=rD2)

# run simulation
simulation = simul.Simulation(layer_system=lay_sys, particle_list=part_list, initial_field=planewave)
simulation2 = simul.Simulation(layer_system=lay_sys, particle_list=part_list2, initial_field=planewave2)
simulation.run()
simulation2.run()
scattered_ff = scf.scattered_far_field(ld, simulation.particle_list, simulation.layer_system)
scattered_ff_2 = scf.scattered_far_field(ld, simulation2.particle_list, simulation2.layer_system)


def test_t_matrix_rotation():
    err = (sum(scattered_ff.integral()) - sum(scattered_ff_2.integral())) / sum(scattered_ff.integral())
    print('error t_matrix_rotation', err)
    assert err < 1e-4


if __name__ == '__main__':
    test_rotation()
    test_t_matrix_rotation()
