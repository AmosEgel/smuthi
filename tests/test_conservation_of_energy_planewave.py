# -*- coding: utf-8 -*-
"""This script runs a simulation for a single sphere on a substrate, illuminated by a plane wave."""

import numpy as np
import smuthi.linear_system as lin
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.index_conversion as idx
import smuthi.t_matrix as tmt
import smuthi.particle_coupling as coup
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.post_processing as pp
import matplotlib.pyplot as plt


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

idx.set_swe_specs(l_max=lmax)

# initialize particle object
part_col = part.ParticleCollection()
part_col.add_sphere(110, 2.4 + 0.0j, [100, 100, 150])
part_col.add_sphere(120, 1.9 + 0.0j, [-100, -100, 250])
part_col.add_sphere(90, 1.7 + 0.0j, [-200, 100, 300])

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [2, 1.3, 2])

# initialize initial field object
init_fld = init.InitialFieldCollection(vacuum_wavelength=vacuum_wavelength)
init_fld.add_planewave(amplitude=plane_wave_amplitude, polar_angle=plane_wave_polar_angle,
                       azimuthal_angle=plane_wave_azimuthal_angle, polarization=plane_wave_polarization,
                       reference_point=[0, 0, 400])

# initialize linear system object
lin_sys = lin.LinearSystem()

# initialize simulation object
simulation = simul.Simulation(lay_sys, part_col, init_fld,
                              wr_neff_contour=coord.ComplexContour(neff_waypoints, neff_discr))
simulation.run()

scs = pp.scattering_cross_section(initial_field_collection=simulation.initial_field_collection,
                                  particle_collection=simulation.particle_collection,
                                  linear_system=simulation.linear_system,
                                  layer_system=simulation.layer_system)

ecs = pp.extinction_cross_section(initial_field_collection=simulation.initial_field_collection,
                                  particle_collection=simulation.particle_collection,
                                  linear_system=simulation.linear_system,
                                  layer_system=simulation.layer_system)


def test_optical_theorem():
    relerr = abs((scs['total'][0] + scs['total'][1] + ecs['forward'] + ecs['backward'])
                 / (scs['total'][0] + scs['total'][1]))
    assert relerr < 1e-4


if __name__ == '__main__':
    test_optical_theorem()
