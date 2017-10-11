import smuthi.initial_field as init
import smuthi.particles as part
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.layers as lay
import smuthi.scattered_field as sf
import numpy as np


ld = 550
rD1 = [100, -100, 130]
D1 = [1e7, 2e7, 3e7]
rD2 = [-100, 100, 70]
D2 = [3e7, -2e7, 1e7]
rD3 = [-100, 100, -100]
D3 = [-2e7, 3e7, 1e7]

# waypoints = [0, 0.8, 0.8-0.1j, 2.1-0.1j, 2.1, 3]
neff_max = 3
neff_discr = 2e-2

coord.set_default_k_parallel(vacuum_wavelength = ld, neff_resolution=neff_discr, neff_max=neff_max)

# initialize particle object
sphere1 = part.Sphere(position=[200, 200, 300], refractive_index=2.4 + 0.0j, radius=110, l_max=3, m_max=3)
sphere2 = part.Sphere(position=[-200, -200, 300], refractive_index=2.4 + 0.0j, radius=120, l_max=3, m_max=3)
sphere3 = part.Sphere(position=[-200, 200, 300], refractive_index=2.5 + 0.0j, radius=90, l_max=3, m_max=3)
part_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [2, 1.3, 2])

# initialize dipole object
dipole1 = init.DipoleSource(vacuum_wavelength=ld, dipole_moment=D1, position=rD1)

dipole2 = init.DipoleSource(vacuum_wavelength=ld, dipole_moment=D2, position=rD2)

dipole3 = init.DipoleSource(vacuum_wavelength=ld, dipole_moment=D3, position=rD3)

dipole_collection = init.DipoleCollection(vacuum_wavelength=ld)
dipole_collection.append(dipole1)
dipole_collection.append(dipole2)
dipole_collection.append(dipole3)

# run simulation
simulation = simul.Simulation(layer_system=lay_sys, particle_list=part_list, initial_field=dipole_collection)

simulation.run()

# evaluate power
power_list = simulation.initial_field.dissipated_power(particle_list=simulation.particle_list,
                                                       layer_system=simulation.layer_system)
power = sum(power_list)
ff_tup = sf.total_far_field(simulation.initial_field, simulation.particle_list, simulation.layer_system)


def test_energy_conservation():
    ff_power = sum(ff_tup[0].integral())
    err = abs((power - ff_power) / ff_power)
    print('ff power', ff_power)
    print('diss power list', power_list)
    print('diss power', power)
    print('error', err)
    assert err < 1e-4


if __name__ == '__main__':
    test_energy_conservation()
