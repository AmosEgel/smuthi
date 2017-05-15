# -*- coding: utf-8 -*-
"""
This script computes the differential cross section (DSC) of a spheroid on a substrate.
"""

import smuthi.index_conversion as idx
import smuthi.simulation as simul
import smuthi.post_processing as pp
import os
import matplotlib.pyplot as plt
import numpy as np

# Parameter input ----------------------------
vacuum_wavelength = 0.550

ambient_refractive_index = 1
substrate_refractive_index = 1.52

particle_refractive_index = 1.6
semi_axis_c = 0.050
semi_axis_a = 0.200
distance_substrate_particle = 0

incoming_wave_polar_angle = np.pi
incoming_wave_azimuthal_angle = 0
incoming_wave_polarization = 0

dsc_polar_angles = np.linspace(0, 89, 90) * np.pi / 180
dsc_azimuthal_angles = [0, np.pi / 2]

lmax = 10
neff_max = 2
dneff = 1e-2
nrank = 10
# --------------------------------------------
idx.set_swe_specs(l_max=lmax)
simulation = simul.Simulation()
simulation.tmatrix_method = {'algorithm': 'nfm-ds', 'use discrete sources': True, 'nint': 200, 'nrank': nrank}
simulation.wr_neff_contour.neff_waypoints = [0, 0.5, 0.5 - 0.05j, 2 - 0.05j, 2, neff_max]
simulation.wr_neff_contour.neff_discretization = dneff
simulation.particle_collection.add_spheroid(semi_axis_c=semi_axis_c, semi_axis_a=semi_axis_a,
                                            refractive_index=particle_refractive_index,
                                            position=[0, 0, semi_axis_c + distance_substrate_particle])

simulation.initial_field_collection.add_planewave(polar_angle=incoming_wave_polar_angle,
                                                  azimuthal_angle=incoming_wave_azimuthal_angle,
                                                  polarization=incoming_wave_polarization)
simulation.initial_field_collection.vacuum_wavelength = vacuum_wavelength

simulation.layer_system.refractive_indices = [substrate_refractive_index, ambient_refractive_index]
simulation.layer_system.thicknesses = [0, 0]

simulation.post_processing.tasks = [{'task': 'evaluate cross sections', 'polar angles': np.linspace(0, np.pi, 500),
                                     'azimuthal angles': np.linspace(0, 2* np.pi, 500)}]


os.chdir('..')
simulation.run()
dsc = simulation.post_processing.scattering_cross_section
esc = simulation.post_processing.extinction_cross_section
print('conservation of energy error:',
      abs(dsc['total'][0] + dsc['total'][1] - esc['top'] - esc['bottom']) / abs(dsc['total'][0] + dsc['total'][1]))
scs = pp.scattering_cross_section(initial_field_collection=simulation.initial_field_collection,
                                  azimuthal_angles=dsc_azimuthal_angles,
                                  polar_angles=dsc_polar_angles,
                                  particle_collection=simulation.particle_collection,
                                  linear_system=simulation.linear_system,
                                  layer_system=simulation.layer_system)


# compare to DSC results:
dsc00_data = np.loadtxt('data/DSC00.dat', skiprows=2)
ind_sel = (dsc00_data[:, 0] >= 0)

plt.figure()
plt.semilogy(scs['polar angles'] * 180 / np.pi, scs['differential'][0, :, 0] + scs['differential'][1, :, 0],
             label='TE, smuthi')
plt.semilogy(scs['polar angles'] * 180 / np.pi, scs['differential'][0, :, 1] + scs['differential'][1, :, 1],
             label='TM, smuthi')

plt.semilogy(dsc00_data[ind_sel, 0], dsc00_data[ind_sel, 1], '.', label='TE, DS')
plt.semilogy(dsc00_data[ind_sel, 0], dsc00_data[ind_sel, 2], '.', label='TM, DS')
plt.legend()

plt.xlabel('polar angle (degree)')
plt.ylabel('DCS in micron^2')
plt.title('Dielectric spheroid (n=1.6) on dielectric substrate (n=1.52)')


plt.show()
