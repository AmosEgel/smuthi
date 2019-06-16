import smuthi.initial_field as init
import smuthi.particles as part
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.layers as lay
import smuthi.scattered_field as sf
import numpy as np

ld = 550
rD = [100, -100, 100]
D = [1e7, 2e7, 3e7]
waypoints = [0, 0.8, 0.8-0.1j, 2.1-0.1j, 2.1, 4]
neff_discr = 2e-2

coord.set_default_k_parallel(vacuum_wavelength = ld, neff_waypoints=waypoints, neff_resolution=neff_discr)

# initialize particle object
sphere1 = part.Sphere(position=[200, 200, 300], refractive_index=2.4 + 0.0j, radius=110, l_max=3, m_max=3)
sphere2 = part.Sphere(position=[-200, -200, 300], refractive_index=2.4 + 0.0j, radius=120, l_max=3, m_max=3)
sphere3 = part.Sphere(position=[-200, 200, 300], refractive_index=2.5 + 0.0j, radius=90, l_max=3, m_max=3)
part_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [2, 1.3, 2])

# initialize dipole object
dipole = init.DipoleSource(vacuum_wavelength=ld, dipole_moment=D, position=rD)

# run simulation
simulation = simul.Simulation(layer_system=lay_sys, particle_list=part_list, initial_field=dipole, log_to_terminal=False)
simulation.run()

power_hom = dipole.dissipated_power_homogeneous_background(layer_system=simulation.layer_system)
power = dipole.dissipated_power(particle_list=simulation.particle_list, layer_system=simulation.layer_system)
power2 = dipole.dissipated_power_alternative(particle_list=simulation.particle_list, layer_system=simulation.layer_system)
ff_tup = sf.total_far_field(simulation.initial_field, simulation.particle_list, simulation.layer_system)


def test_energy_conservation():
    ff_power = sum(ff_tup[0].integral())
    err = abs((power - ff_power) / ff_power)
    print('ff power', ff_power)
    print('diss power', power)
    print('diss power old', power2)
    print('hom power', power_hom)
    print('error', err)
    assert err < 1e-4
    print("Test passed.")


def test_power_prototype():
    diss_pow_tspl = 8.5902975e+05
    print('diss power tspl', diss_pow_tspl)
    err = abs((power - diss_pow_tspl) / diss_pow_tspl)
    print('error', err)
    assert err < 1e-4
    print("Test passed.")

if __name__ == '__main__':
    test_energy_conservation()
    test_power_prototype()
