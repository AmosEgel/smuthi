# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 08:57:59 2018

@author: theobald2
"""

# -*- coding: utf-8 -*-
from imp import reload
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.simulation as simul
import smuthi.coordinates as coord
import nearfield_spheroid as nearsph
import smuthi.scattered_field as scf
import smuthi.field_expansion as fldex
import smuthi.graphical_output as graph
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import tempfile
import shutil
import imageio


reload(nearsph)


wl = 550
waypoints = [0, 0.8, 0.8-0.1j, 2.1-0.1j, 2.1, 3]
neff_discr = 2e-2
coord.set_default_k_parallel(vacuum_wavelength = wl, neff_waypoints=waypoints, neff_resolution=neff_discr)
# initialize layer system
lay_sys = lay.LayerSystem([0, 1600, 0], [1, 1, 1])
# initialize particle list
spheroid1 = part.Spheroid(position=[0, 0, 800], euler_angles=[0, 0.25 * np.pi, 0],
                          refractive_index=2.4 + 0.0j, semi_axis_c=25, semi_axis_a=100, l_max=15, m_max=15)
#spheroid2 = part.Spheroid(position=[-150, 0, 200], euler_angles=[1.2, 0.1, 0],
#                          refractive_index=2.4 + 0.0j, semi_axis_c=25, semi_axis_a=100, l_max=5, m_max=5)
part_list = [spheroid1]


# initialize plane wave object
rD = [0, 0, 0]
planewave = init.PlaneWave(vacuum_wavelength=wl, polar_angle=np.pi, azimuthal_angle=0, polarization=0, amplitude=1,
                           reference_point=rD)

# run simulation
simulation = simul.Simulation(layer_system=lay_sys, particle_list=part_list, initial_field=planewave)
simulation.run()

fl, Ey, fp0, dim1vec, dim2vec = nearsph.pwe_nearfield_superposition(-200,200,0.001,0.001,600,1000,5,k_parallel='default', azimuthal_angles='default',
                                                      simulation=simulation)
grid_x = np.meshgrid(dim1vec, dim2vec)

comsol_data_real = np.loadtxt('data/Comsol/1Ellipsoids_sp_pw_n=1_2,4_100_100_25_beta45_xzplane_real.txt', comments='%')
comsol_data_imag = np.loadtxt('data/Comsol/1Ellipsoids_sp_pw_n=1_2,4_100_100_25_beta45_xzplane_imag.txt', comments='%')
comsol_real_ey = interp.griddata(comsol_data_real[:, :2], comsol_data_real[:, 2], 
                                 (grid_x[0] * 1e-9, grid_x[1] * 1e-9), method='linear')

comsol_imag_ey = interp.griddata(comsol_data_imag[:, :2], comsol_data_imag[:, 2], 
                                 (grid_x[0] * 1e-9, grid_x[1] * 1e-9), method='linear')

vmin = -0.5
vmax = 0.5

#plt.figure()
##graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
##graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
#plt.gca().set_aspect("equal")
#plt.pcolormesh(dim1vec, dim2vec, Ey.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
#plt.colorbar()
#
#plt.figure()
##graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
##graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
#plt.gca().set_aspect("equal")
#plt.pcolormesh(dim1vec, dim2vec, comsol_real_ey, vmin=vmin, vmax=vmax, cmap='RdYlBu')
#plt.colorbar()
#
#plt.figure()
##graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
##graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
#plt.gca().set_aspect("equal")
#plt.pcolormesh(dim1vec, dim2vec, Ey.imag, vmin=vmin, vmax=vmax, cmap='RdYlBu')
#plt.colorbar()
#
#plt.figure()
#graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
##graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
#plt.gca().set_aspect("equal")
#plt.pcolormesh(dim1vec, dim2vec, comsol_imag_ey, vmin=vmin, vmax=vmax, cmap='RdYlBu')
#plt.colorbar()


comsol_dat = comsol_real_ey + 1j * comsol_imag_ey
diff = Ey - comsol_dat 
difference = np.sqrt(diff.real**2 + diff.imag**2) / np.sqrt(comsol_dat.real**2 + comsol_dat.imag**2)


# field for SWE only
#fp0, dim1vec, dim2vec = nearsph.fieldpoints(-200,200,0.001,0.001,600,1000,5)
temp_array = np.array([0, 0, 0, 0, 0], dtype=float)[None,:]
for k in range(np.size(fp0[:, 0, 0])):
    for l in range(np.size(fp0[0, :, 0])):
        temp_array = np.append(temp_array, np.array([k , l, fp0[k, l, 0], fp0[k, l, 1], fp0[k, l, 2]])[None,:],
                               axis=0)
temp_array = np.delete(temp_array, 0, 0)
ex_con, ey_con, ez_con = (simulation.particle_list[0].scattered_field.electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0],
                          simulation.particle_list[0].scattered_field.electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1], 
                          simulation.particle_list[0].scattered_field.electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2])
ey_con = np.reshape(ey_con, [dim1vec.size, dim2vec.size], order='C')
diff_con = ey_con - comsol_dat 
difference_con = np.sqrt(diff_con.real**2 + diff_con.imag**2) / np.sqrt(comsol_dat.real**2 + comsol_dat.imag**2)

fp_inside = nearsph.inside_particles(fp0, simulation.particle_list)
fp_inside = np.array(fp_inside[:,:,3], dtype=bool)

fig, ax = plt.subplots()
#graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
#graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
plt.gca().set_aspect("equal")
plt.pcolormesh(dim1vec, dim2vec, difference, vmin=0, vmax=0.1, cmap='RdYlBu')
ax.contour(dim1vec, dim2vec, fp_inside, colors='k')
plt.colorbar()



fig, ax = plt.subplots()
#graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
#graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
plt.gca().set_aspect("equal")
plt.pcolormesh(dim1vec, dim2vec, difference_con, vmin=0, vmax=0.1, cmap='RdYlBu')
ax.contour(dim1vec, dim2vec, fp_inside, colors='k')
plt.colorbar()


