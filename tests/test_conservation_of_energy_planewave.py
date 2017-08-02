# -*- coding: utf-8 -*-
"""This script runs a simulation for a single sphere on a substrate, illuminated by a plane wave."""

import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.far_field as ff


# Parameter input ----------------------------
vacuum_wavelength = 550
plane_wave_polar_angle = np.pi * 7/8
plane_wave_azimuthal_angle = np.pi * 1/3
plane_wave_polarization = 0
plane_wave_amplitude = 1
lmax = 3
neff_waypoints = [0, 0.5, 0.8-0.01j, 2-0.01j, 2.5, 5]
neff_discr = 5e-3

# --------------------------------------------

# initialize particle object
sphere1 = part.Sphere(position=[100, 100, 150], refractive_index=2.4 + 0.0j, radius=110, l_max=lmax, m_max=lmax)
sphere2 = part.Sphere(position=[-100, -100, 250], refractive_index=1.9 + 0.0j, radius=120, l_max=lmax, m_max=lmax)
sphere3 = part.Sphere(position=[-200, 100, 300], refractive_index=1.7 + 0.0j, radius=90, l_max=lmax, m_max=lmax)
particle_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [2, 1.3, 2])

# initialize initial field object
init_fld = init.PlaneWave(vacuum_wavelength=vacuum_wavelength, polar_angle=plane_wave_polar_angle,
                          azimuthal_angle=plane_wave_azimuthal_angle, polarization=plane_wave_polarization,
                          amplitude=plane_wave_amplitude, reference_point=[0, 0, 400])

# initialize simulation object
simulation = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld,
                              wr_neff_contour=coord.ComplexContour(neff_waypoints, neff_discr))
simulation.run()

scs = ff.scattering_cross_section(initial_field=simulation.initial_field, particle_list=simulation.particle_list,
                                  layer_system=simulation.layer_system)

ecs = ff.extinction_cross_section(initial_field=simulation.initial_field,particle_list=simulation.particle_list,
                                  layer_system=simulation.layer_system)


def test_optical_theorem():
    relerr = abs((scs['total'][0] + scs['total'][1] - ecs['top'] - ecs['bottom'])
                 / (scs['total'][0] + scs['total'][1]))
    # print(scs['total'][0] + scs['total'][1])
    # print(ecs['top'] + ecs['bottom'])
    # print(relerr)
    assert relerr < 1e-4


if __name__ == '__main__':
    test_optical_theorem()
