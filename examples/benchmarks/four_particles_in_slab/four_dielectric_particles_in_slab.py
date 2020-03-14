#*****************************************************************************#
# This benchmark compares the electric field of a dipole source in a slab     #
# waveguide on a metallic substrate including four particles with different   #
# shapes as computed with Smuthi to that computed with COMSOL (FEM).          #
# The script runs with Smuthi version 0.8.6                                   #
#*****************************************************************************#

import smuthi.initial_field as init
import smuthi.particles as part
import smuthi.simulation as simul
import smuthi.layers as lay
import smuthi.postprocessing.scattered_field as sf
import smuthi.utility.cuda as cu
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import os

# Enable GPU usage. Uncomment if you receive GPU related errors
cu.enable_gpu()

vacuum_wavelength = 550

# Set the multipole truncation order
# We invite the user to play with this parameter to see how it affects
# accuracy and runtime.
lmax = 5

# particles
sphere = part.Sphere(position=[300, 300, 250], 
                     refractive_index=3,
                     radius=120, 
                     l_max=lmax)

cylinder = part.FiniteCylinder(position=[300, -300, 250], 
                               refractive_index=3,
                               cylinder_radius=100, 
                               cylinder_height=200,
                               l_max=lmax)

prolate = part.Spheroid(position=[-300,300,250],
                        refractive_index=3,
                        semi_axis_a=80,
                        semi_axis_c=150,
                        l_max=lmax)

oblate = part.Spheroid(position=[-300,-300,250],
                       refractive_index=3,
                       semi_axis_a=150,
                       semi_axis_c=80,
                       l_max=lmax)

four_particles = [sphere, cylinder, prolate, oblate]

# layer system
three_layers = lay.LayerSystem(thicknesses=[0, 500, 0],
                               refractive_indices=[1+6j, 2, 1.5])

# dipole source
dipole = init.DipoleSource(vacuum_wavelength=vacuum_wavelength,
                           dipole_moment=[1, 0, 0], 
                           position=[0, 0, 250])

# run simulation
simulation = simul.Simulation(layer_system=three_layers, 
                              particle_list=four_particles, 
                              initial_field=dipole)
simulation.run()

# near field
scat_fld_exp = sf.scattered_field_piecewise_expansion(vacuum_wavelength, 
                                                      four_particles, 
                                                      three_layers)

# field along probing line ----------------------------------------------------
x_line = np.arange(-1500, 1501, 10)
y_line = x_line - x_line
z_line = y_line + 750
e_scat = scat_fld_exp.electric_field(x=x_line, y=y_line, z=z_line) 

e_init = simulation.initial_field.electric_field(x=x_line,
                                                 y=y_line,
                                                 z=z_line,
                                                 layer_system=three_layers)

# load COMSOL results ---------------------------------------------------------
filename = (os.path.abspath(os.path.dirname(__file__))
            + "/comsol_data_along_line_four_dielectric_particles_in_slab.txt")
comsol_data = np.loadtxt(filename, comments='%')

comsol_x = comsol_data[:, 0] * 1e9  # the factor of 1e9 is for nm

# the complex conjugate is beacuse comsol works with e^(iwt-ikx) convention
# the numeric scale factor is because in comsol, the dipole source is
# specified in different physical units
comsol_ey = (-1j * comsol_data[:, 5] - comsol_data[:, 6]) * 3e-23   
comsol_real_ey = interp.interp1d(comsol_x, comsol_ey.real)
comsol_imag_ey = interp.interp1d(comsol_x, comsol_ey.imag)

# plot ------------------------------------------------------------------------
x_coarse = np.arange(x_line[0], x_line[-1], 50)
plt.plot(x_coarse, comsol_real_ey(x_coarse), 'ob')
plt.plot(x_coarse, comsol_imag_ey(x_coarse), 'or')
plt.plot(x_line, (e_scat[1] + e_init[1]).real, 'b')
plt.plot(x_line, (e_scat[1] + e_init[1]).imag, '--r')
plt.xlabel('x (nm)')
plt.ylabel(r'$E_y$ (a.u.)')
plt.xlim((x_line.min(), x_line.max()))
plt.ylim((-2e-9, 1.5e-9))
plt.legend(('real part, comsol', 
            'imaginary part, comsol', 
            'real part, smuthi, lmax=%i'%lmax, 
            'imaginary part, smuthi, lmax=%i'%lmax))
plt.title('field along probing line')

# field along probing plane ---------------------------------------------------
resolution = 100  # decrease for faster runtime
x_plane, y_plane = np.mgrid[-1000:1000:(resolution*1j), 
                            -1000:1000:(resolution*1j)]
xy_plane = np.zeros((x_plane.size, 2))
xy_plane[:,0] = x_plane.reshape(-1)
xy_plane[:,1] = y_plane.reshape(-1)
 
e_scat = scat_fld_exp.electric_field(x=x_plane,
                                     y=y_plane, 
                                     z=x_plane-x_plane+750) 
 
e_init = simulation.initial_field.electric_field(x=x_plane, 
                                                 y=y_plane, 
                                                 z=x_plane-x_plane+750, 
                                                 layer_system=three_layers)

# load COMSOL results ---------------------------------------------------------
filename = (os.path.abspath(os.path.dirname(__file__))
            + "/comsol_data_along_plane_four_dielectric_particles_in_slab.txt")
comsol_data = np.loadtxt(filename, comments='%')

comsol_xy = comsol_data[:, :2] * 1e9  # the factor of 1e9 is for nm

# the complex conjugate is beacuse comsol works with e^(iwt-ikx) convention
# the numeric scale factor is because in comsol, the dipole source is
# specified in different physical units
comsol_ey = (-1j * comsol_data[:, 3] - comsol_data[:, 4]) * 3e-23   

comsol_real_ey = interp.griddata(comsol_xy, 
                                 comsol_ey.real, 
                                 (x_plane, y_plane), 
                                 method='linear')

comsol_imag_ey = interp.griddata(comsol_xy, 
                                 comsol_ey.imag, 
                                 (x_plane, y_plane), 
                                 method='linear')

# plot ------------------------------------------------------------------------
vmin = -7e-9
vmax = 7e-9

plt.figure()
plt.pcolormesh(x_plane, y_plane, e_scat[1].imag + e_init[1].imag, 
               vmin=vmin, vmax=vmax, cmap='RdYlBu')
plt.xlim((x_plane.min(), x_plane.max()))
plt.ylim((y_plane.min(), y_plane.max()))
plt.axes().set_aspect('equal')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.title(r'imag E_y along probing plane, Smuthi')

plt.figure()
plt.pcolormesh(x_plane, y_plane, comsol_imag_ey.reshape(x_plane.shape), 
               vmin=vmin, vmax=vmax, cmap='RdYlBu')
plt.xlim((x_plane.min(), x_plane.max()))
plt.ylim((y_plane.min(), y_plane.max()))
plt.axes().set_aspect('equal')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.title(r'imag E_y along probing plane, Comsol')

plt.figure()
plt.pcolormesh(x_plane, y_plane, e_scat[1].real + e_init[1].real, 
               vmin=vmin, vmax=vmax, cmap='RdYlBu')
plt.xlim((x_plane.min(), x_plane.max()))
plt.ylim((y_plane.min(), y_plane.max()))
plt.axes().set_aspect('equal')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.title(r'real E_y along probing plane, Smuthi')


plt.figure()
plt.pcolormesh(x_plane, y_plane, comsol_real_ey.reshape(x_plane.shape), 
               vmin=vmin, vmax=vmax, cmap='RdYlBu')
plt.xlim((x_plane.min(), x_plane.max()))
plt.ylim((y_plane.min(), y_plane.max()))
plt.axes().set_aspect('equal')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.title(r'real E_y along probing plane, Comsol')

plt.show()
