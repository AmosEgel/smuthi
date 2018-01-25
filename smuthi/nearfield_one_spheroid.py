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
spheroid1 = part.Spheroid(position=[0, 0, 800], euler_angles=[0, 0, 0],
                          refractive_index=2.4 + 0.0j, semi_axis_c=25, semi_axis_a=100, l_max=5, m_max=5)
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

fl, Ey, dim1vec, dim2vec = nearsph.pwe_nearfield_superposition(-400,400,0.001,0.001,400,1200,10,k_parallel='default', azimuthal_angles='default',
                                                      simulation=simulation)
grid_x = np.meshgrid(dim1vec, dim2vec)

comsol_data_real = np.loadtxt('data/Comsol/1Ellipsoids_sp_pw_n=1_2,4_100_100_25_xzplane_real.txt', comments='%')
comsol_data_imag = np.loadtxt('data/Comsol/1Ellipsoids_sp_pw_n=1_2,4_100_100_25_xzplane_imag.txt', comments='%')
comsol_real_ey = interp.griddata(comsol_data_real[:, :2], comsol_data_real[:, 2], 
                                 (grid_x[0] * 1e-9, grid_x[1] * 1e-9), method='linear')

comsol_imag_ey = interp.griddata(comsol_data_imag[:, :2], comsol_data_imag[:, 2], 
                                 (grid_x[0] * 1e-9, grid_x[1] * 1e-9), method='linear')

vmin = -0.5
vmax = 0.5

#plt.figure()
#graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
##graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
#plt.gca().set_aspect("equal")
#plt.pcolormesh(dim1vec, dim2vec, Ey.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
#plt.colorbar()
#
#plt.figure()
#graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
##graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
#plt.gca().set_aspect("equal")
#plt.pcolormesh(dim1vec, dim2vec, comsol_real_ey, vmin=vmin, vmax=vmax, cmap='RdYlBu')
#plt.colorbar()
#
#plt.figure()
#graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
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

plt.figure()
graph.plot_particles(-400,400,0.001,0.001,400,1200, simulation.particle_list, np.Inf)
#graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
plt.gca().set_aspect("equal")
plt.pcolormesh(dim1vec, dim2vec, difference, vmin=0, vmax=0.5, cmap='RdYlBu')
plt.colorbar()



#b_particle = part_list[0].scattered_field.coefficients
#pwe_particle1 = fldex.swe_to_pwe_conversion(swe=part_list[0].scattered_field, k_parallel='default', azimuthal_angles='default', layer_system=lay_sys,
#                                        layer_number=1, layer_system_mediated=False)
#
#
#
#scatt_fields = ['E_x_scat', 'E_y_scat', 'E_z_scat']
## from graphical_output.show_near_field
#quantities_to_plot=scatt_fields
#save_plots=False,
#show_plots=True
#save_animations=False
#save_data=False
#outputdir='.'
#xmin, xmax, ymin, ymax, zmin, zmax =-400, 400, 0.01, 0.01, 0, 800
#resolution=20
#interpolate=10
#k_parallel='default'
#azimuthal_angles='default'
#simulation=simulation
#max_field=0.5
#max_particle_distance=float('inf')
#
#
#vacuum_wavelength = simulation.initial_field.vacuum_wavelength
#if xmin == xmax:
#    dim1vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
#    dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
#    yarr, zarr = np.meshgrid(dim1vec, dim2vec)
#    xarr = yarr - yarr + xmin
#    dim1name = 'y (' + simulation.length_unit + ')'
#    dim2name = 'z (' + simulation.length_unit + ')'
#elif ymin == ymax:
#    dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
#    dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
#    xarr, zarr = np.meshgrid(dim1vec, dim2vec)
#    yarr = xarr - xarr + ymin
#    dim1name = 'x (' + simulation.length_unit + ')'
#    dim2name = 'z (' + simulation.length_unit + ')'
#else:
#    dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
#    dim2vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
#    xarr, yarr = np.meshgrid(dim1vec, dim2vec)
#    zarr = xarr - xarr + zmin
#    dim1name = 'x (' + simulation.length_unit + ')'
#    dim2name = 'y (' + simulation.length_unit + ')'
#
#scat_fld_exp_up = pwe_particle1[0]        
#scat_fld_exp_down = pwe_particle1[1]
#e_x_scat_raw, e_y_scat_raw, e_z_scat_raw = (scat_fld_exp_up.electric_field(xarr, yarr, zarr)[0] + scat_fld_exp_down.electric_field(xarr, yarr, zarr)[0],
#                                            scat_fld_exp_up.electric_field(xarr, yarr, zarr)[1] + scat_fld_exp_down.electric_field(xarr, yarr, zarr)[1],
#                                            scat_fld_exp_up.electric_field(xarr, yarr, zarr)[2] + scat_fld_exp_down.electric_field(xarr, yarr, zarr)[2])
#    
#e_x_init_raw, e_y_init_raw, e_z_init_raw = simulation.initial_field.electric_field(xarr, yarr, zarr, simulation.layer_system)
#
#if True == True:
#    if interpolate is None:
#        e_x_scat, e_y_scat, e_z_scat = e_x_scat_raw, e_y_scat_raw, e_z_scat_raw
#        e_x_init, e_y_init, e_z_init = e_x_init_raw, e_y_init_raw, e_z_init_raw
#        dim1vecfine = dim1vec
#        dim2vecfine = dim2vec
#    else:
#        dim1vecfine = np.linspace(dim1vec[0], dim1vec[-1], (dim1vec[-1] - dim1vec[0]) / interpolate + 1, endpoint=True)
#        dim2vecfine = np.linspace(dim2vec[0], dim2vec[-1], (dim2vec[-1] - dim2vec[0]) / interpolate + 1, endpoint=True)
#
#        real_ex_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_scat_raw.real)
#        imag_ex_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_scat_raw.imag)
#        e_x_scat = (real_ex_scat_interpolant(dim2vecfine, dim1vecfine)
#                    + 1j * imag_ex_scat_interpolant(dim2vecfine, dim1vecfine))
#
#        real_ey_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_scat_raw.real)
#        imag_ey_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_scat_raw.imag)
#        e_y_scat = (real_ey_scat_interpolant(dim2vecfine, dim1vecfine)
#                    + 1j * imag_ey_scat_interpolant(dim2vecfine, dim1vecfine))
#
#        real_ez_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_scat_raw.real)
#        imag_ez_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_scat_raw.imag)
#        e_z_scat = (real_ez_scat_interpolant(dim2vecfine, dim1vecfine)
#                    + 1j * imag_ez_scat_interpolant(dim2vecfine, dim1vecfine))
#
#        real_ex_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_init_raw.real)
#        imag_ex_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_init_raw.imag)
#        e_x_init = (real_ex_init_interpolant(dim2vecfine, dim1vecfine)
#                    + 1j * imag_ex_init_interpolant(dim2vecfine, dim1vecfine))
#
#        real_ey_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_init_raw.real)
#        imag_ey_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_init_raw.imag)
#        e_y_init = (real_ey_init_interpolant(dim2vecfine, dim1vecfine)
#                    + 1j * imag_ey_init_interpolant(dim2vecfine, dim1vecfine))
#
#        real_ez_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_init_raw.real)
#        imag_ez_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_init_raw.imag)
#        e_z_init = (real_ez_init_interpolant(dim2vecfine, dim1vecfine)
#                    + 1j * imag_ez_init_interpolant(dim2vecfine, dim1vecfine))
#
#    if max_field is None:
#        vmin = None
#        vmax = None
#    else:
#        vmin = -max_field
#        vmax = max_field
#
#    for jq, quantity in enumerate(quantities_to_plot):
#
#        filename = 'E'
#
#        fig = plt.figure()
#        if 'scat' in quantity:
#            e_x, e_y, e_z = e_x_scat, e_y_scat, e_z_scat
#            field_type_string = 'scattered electric field'
#            filename = filename + '_scat'
#        elif 'init' in quantity:
#            e_x, e_y, e_z = e_x_init, e_y_init, e_z_init
#            field_type_string = 'initial electric field'
#            filename = filename + '_init'
#        else:
#            e_x, e_y, e_z = e_x_scat + e_x_init, e_y_scat + e_y_init, e_z_scat + e_z_init
#            field_type_string = 'total electric field'
#
#        if 'norm' in quantity:
#            e = np.sqrt(abs(e_x)**2 + abs(e_y)**2 + abs(e_z)**2)
#            filename = 'norm_' + filename
#            plt.pcolormesh(dim1vecfine, dim2vecfine, np.sqrt(abs(e_x)**2 + abs(e_y)**2 + abs(e_z)**2), vmin=0,
#                           vmax=vmax, cmap='inferno')
#            plt_title = 'norm of ' + field_type_string
#            plt.title(plt_title)
#        else:
#            if '_x' in quantity:
#                e = e_x
#                filename = filename + '_x'
#                plt_title = 'x-component of ' + field_type_string
#            elif '_y' in quantity:
#                e = e_y
#                filename = filename + '_y'
#                plt_title = 'y-component of ' + field_type_string
#            elif '_z' in quantity:
#                e = e_z
#                filename = filename + '_z'
#                plt_title = 'z-component of ' + field_type_string
#            plt.pcolormesh(dim1vecfine, dim2vecfine, e.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
#            plt.title(plt_title)
#
#        plt.colorbar()
#        plt.xlabel(dim1name)
#        plt.ylabel(dim2name)
#
#        if not zmin == zmax:
#            graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
#
#        graph.plot_particles(xmin, xmax, ymin, ymax, zmin, zmax, simulation.particle_list, max_particle_distance)
#
#        plt.gca().set_aspect("equal")
#
#        export_filename = outputdir + '/' + filename
#        if save_plots:
#            plt.savefig(export_filename + '.png')
#        if save_animations:
#            if not 'norm' in quantity:
#                tempdir = tempfile.mkdtemp()
#                images = []
#                for i_t, t in enumerate(np.linspace(0, 1, 20, endpoint=False)):
#                    tempfig = plt.figure()
#                    e_t = e * np.exp(-1j * t * 2 * np.pi)
#                    plt.pcolormesh(dim1vecfine, dim2vecfine, e_t.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
#                    plt.title(plt_title)
#                    plt.colorbar()
#                    plt.xlabel(dim1name)
#                    plt.ylabel(dim2name)
#                    if not zmin == zmax:
#                        graph.plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
#                    graph.plot_particles(xmin, xmax, ymin, ymax, zmin, zmax, simulation.particle_list, max_particle_distance)
#                    plt.gca().set_aspect("equal")
#                    tempfig_filename = tempdir + '/temp_' + str(i_t) + '.png'
#                    plt.savefig(tempfig_filename)
#                    plt.close(tempfig)
#                    images.append(imageio.imread(tempfig_filename))
#                imageio.mimsave(export_filename + '.gif', images, duration=0.1)
#                shutil.rmtree(tempdir)
#        if show_plots:
#            plt.draw()
#        else:
#            plt.close(fig)
#
#    if save_data:
#        filename = outputdir + '/spatial_coordinates_along_first_dimension.dat'
#        header = dim1name
#        np.savetxt(filename, dim1vec, header=header)
#
#        filename = outputdir + '/spatial_coordinates_along_second_dimension.dat'
#        header = dim2name
#        np.savetxt(filename, dim2vec, header=header)
#
#        filename = outputdir + '/real_e_init_x.dat'
#        header = 'Real part of x-component of initial electric field.'
#        np.savetxt(filename, e_x_init_raw.real, header=header)
#
#        filename = outputdir + '/imag_e_init_x.dat'
#        header = 'Imaginary part of x-component of initial electric field.'
#        np.savetxt(filename, e_x_init_raw.imag, header=header)
#
#        filename = outputdir + '/real_e_init_y.dat'
#        header = 'Real part of y-component of initial electric field.'
#        np.savetxt(filename, e_y_init_raw.real, header=header)
#
#        filename = outputdir + '/imag_e_init_y.dat'
#        header = 'Imaginary part of y-component of initial electric field.'
#        np.savetxt(filename, e_y_init_raw.imag, header=header)
#
#        filename = outputdir + '/real_e_init_z.dat'
#        header = 'Real part of z-component of initial electric field.'
#        np.savetxt(filename, e_z_init_raw.real, header=header)
#
#        filename = outputdir + '/imag_e_init_z.dat'
#        header = 'Imaginary part of z-component of initial electric field.'
#        np.savetxt(filename, e_z_init_raw.imag, header=header)
#
#        filename = outputdir + '/real_e_scat_x.dat'
#        header = 'Real part of x-component of scattered electric field.'
#        np.savetxt(filename, e_x_scat_raw.real, header=header)
#
#        filename = outputdir + '/imag_e_scat_x.dat'
#        header = 'Imaginary part of x-component of scattered electric field.'
#        np.savetxt(filename, e_x_scat_raw.imag, header=header)
#
#        filename = outputdir + '/real_e_scat_y.dat'
#        header = 'Real part of y-component of scattered electric field.'
#        np.savetxt(filename, e_y_scat_raw.real, header=header)
#
#        filename = outputdir + '/imag_e_scat_y.dat'
#        header = 'Imaginary part of y-component of scattered electric field.'
#        np.savetxt(filename, e_y_scat_raw.imag, header=header)
#
#        filename = outputdir + '/real_e_scat_z.dat'
#        header = 'Real part of z-component of scattered electric field.'
#        np.savetxt(filename, e_z_scat_raw.real, header=header)
#
#        filename = outputdir + '/imag_e_scat_z.dat'
#        header = 'Imaginary part of z-component of scattered electric field.'
#        np.savetxt(filename, e_z_scat_raw.imag, header=header)
#
#
#
#

#
#scatt_fields = ['E_x_scat', 'E_y_scat', 'E_z_scat']
#graph.show_near_field(quantities_to_plot=scatt_fields, save_plots=False, show_plots=True, save_animations=False, save_data=False,
#                    outputdir='.', xmin=-400, xmax=400, ymin=0.01, ymax=0.01, zmin=0, zmax=800, resolution=20, interpolate=None,
#                    k_parallel='default', azimuthal_angles='default', simulation=simulation, max_field=0.5,
#                    max_particle_distance=float('inf'))
#init_fields = ['E_init_x', 'E_init_y', 'E_init_z']
#graph.show_near_field(quantities_to_plot=init_fields, save_plots=False, show_plots=True, save_animations=False, save_data=False,
#                    outputdir='.', xmin=-400, xmax=400, ymin=0.01, ymax=0.01, zmin=0, zmax=800, resolution=20, interpolate=10,
#                    k_parallel='default', azimuthal_angles='default', simulation=simulation, max_field=1,
#                    max_particle_distance=float('inf'))
#
#
#
#
#
#test = scf.scattered_field_piecewise_expansion(vacuum_wavelength=wl, particle_list=part_list, layer_system=lay_sys, k_parallel='default', 
#                                        azimuthal_angles='default', layer_numbers=None)
#pwetest = scf.scattered_field_pwe(vacuum_wavelength=wl, particle_list=part_list, layer_system=lay_sys, layer_number=1, k_parallel='default',
#                        azimuthal_angles='default', include_direct=True, include_layer_response=True)
    

              