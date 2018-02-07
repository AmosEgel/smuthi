# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as interp
import smuthi.coordinates as coord
import smuthi.field_expansion as fldex
import smuthi.scattered_field as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
import tempfile
import shutil
import imageio
import os
import warnings
import sys
            

def plot_layer_interfaces(dim1min, dim1max, layer_system):
    """Add lines to plot to display layer system interfaces

    Args:
        dim1min (float):    From what x-value plot line
        dim1max (float):    To what x-value plot line
        layer_system (smuthi.layers.LayerSystem):   Stratified medium
    """
    for il in range(1, layer_system.number_of_layers()):
        plt.plot([dim1min, dim1max], [layer_system.reference_z(il), layer_system.reference_z(il)], 'g')


def plot_particles(xmin, xmax, ymin, ymax, zmin, zmax, particle_list, max_particle_distance):
    """Add circles, ellipses and rectangles to plot to display spheres, spheroids and cylinders.

    Args:
        xmin (float):   Minimal x-value of plot
        xmax (float):   Maximal x-value of plot
        ymin (float):   Minimal y-value of plot
        ymax (float):   Maximal y-value of plot
        zmin (float):   Minimal z-value of plot
        zmax (float):   Maximal z-value of plot
        particle_list (list): List of smuthi.particles.Particle objects
        max_particle_distance (float):  Plot only particles that ar not further away from image plane
    """
    
    ax = plt.gca()

    if xmin == xmax:
        plane_coord = 0
        draw_coord = [1, 2]
    elif ymin == ymax:
        plane_coord = 1
        draw_coord = [0, 2]
    elif zmin == zmax:
        plane_coord = 2
        draw_coord = [0, 1]
    else:
        raise ValueError('Field points must define a plane')

    for particle in particle_list:
        pos = particle.position
                    
        if abs((xmin, ymin, zmin)[plane_coord] - pos[plane_coord]) > max_particle_distance:
            continue
        
        if type(particle).__name__ == 'Sphere':
            ax.add_patch(Circle((pos[draw_coord[0]], pos[draw_coord[1]]), particle.radius, facecolor='w', 
                                       edgecolor='k'))
        else:
            if not particle.euler_angles == [0, 0, 0]:
                warnings.warn("Drawing rotated particles currently not supported - drawing black disc with size"
                              + " of circumscribing sphere instead")
                ax.add_patch(Circle((pos[draw_coord[0]], pos[draw_coord[1]]), particle.circumscribing_sphere_radius(), 
                                    facecolor='k', edgecolor='k'))
                ax.text(pos[draw_coord[0]], pos[draw_coord[1]], 'rotated ' + type(particle).__name__,
                        verticalalignment='center', horizontalalignment='center', color='blue', fontsize=5)
            else:
                ax.add_patch(Circle((pos[draw_coord[0]], pos[draw_coord[1]]), particle.circumscribing_sphere_radius(), 
                                    linestyle='dashed', facecolor='none', edgecolor='k'))

                if type(particle).__name__ == 'Spheroid':
                    width = 2 * particle.semi_axis_a
                    if plane_coord == 2:
                        height = 2 * particle.semi_axis_a
                    else:
                        height = 2 * particle.semi_axis_c
                    ax.add_patch(Ellipse(xy=(pos[draw_coord[0]], pos[draw_coord[1]]), width=width, height=height,
                                         facecolor='w', edgecolor='k'))

                elif type(particle).__name__ == 'FiniteCylinder':
                    if plane_coord == 2:
                        ax.add_patch(Circle((pos[draw_coord[0]], pos[draw_coord[1]]), particle.cylinder_radius,
                                            facecolor='w', edgecolor='k'))
                    else:
                        ax.add_patch(Rectangle((pos[draw_coord[0]]-particle.cylinder_radius, 
                                                pos[draw_coord[1]]-particle.cylinder_height/2), 
                                               2*particle.cylinder_radius, particle.cylinder_height, facecolor='w', 
                                               edgecolor='k'))


def show_near_field(quantities_to_plot=None, save_plots=False, show_plots=True, save_animations=False, save_data=False,
                    outputdir='.', xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0, resolution_step=25, 
                    interpolate_step=None, interpolation_order = 1, k_parallel='default', azimuthal_angles='default', 
                    simulation=None, max_field=None, min_norm_field=None, max_particle_distance=float('inf')):
    """Plot the electric near field along a plane. To plot along the xy-plane, specify zmin=zmax and so on.

    Args:
        quantities_to_plot: List of strings that specify what to plot. Select from 'E_x', 'E_y', 'E_z', 'norm(E)'
                            The list may contain one or more of the following strings:

                                'E_x'       real part of x-component of complex total electric field
                                'E_y'       real part of y-component of complex total electric field
                                'E_z'       real part of z-component of complex total electric field
                                'norm(E)'   norm of complex total electric field

                                'E_scat_x'       real part of x-component of complex scattered electric field
                                'E_scat_y'       real part of y-component of complex scattered electric field
                                'E_scat_z'       real part of z-component of complex scattered electric field
                                'norm(E_scat)'   norm of complex scattered electric field

                                'E_init_x'       real part of x-component of complex initial electric field
                                'E_init_y'       real part of y-component of complex initial electric field
                                'E_init_z'       real part of z-component of complex initial electric field
                                'norm(E_init)'   norm of complex initial electric field
        save_plots (logical):   If True, plots are exported to file.
        show_plots (logical):   If True, plots are shown
        save_animations (logical):  If True, animated gif-images are exported
        save_data (logical):    If True, raw data are exported to file.
        outputdir (str):        Path to directory where to save the export files
        xmin (float):       Plot from that x (length unit)
        xmax (float):       Plot up to that x (length unit)
        ymin (float):       Plot from that y (length unit)
        ymax (float):       Plot up to that y (length unit)
        zmin (float):       Plot from that z (length unit)
        zmax (float):       Plot up to that z (length unit)
        resolution_step (float):     Compute the field with that spatial resolution (length unit,
                                     distance between computed points)
        interpolate_step (float):    Use spline interpolation with that resolution to plot a smooth
                                     field (length unit, distance between computed points)
        interpolation_order (int):   Splines of that order are used to interpolate. Choose e.g. 1 for linear and 3 for
                                     cubic spline interpolation.                                     
        k_parallel (numpy.ndarray or str):         in-plane wavenumbers for the plane wave expansion
                                                   if 'default', use smuthi.coordinates.default_k_parallel
        azimuthal_angles (numpy.ndarray or str):   azimuthal angles for the plane wave expansion
                                                   if 'default', use smuthi.coordinates.default_azimuthal_angles
        simulation (smuthi.simulation.Simulation):  Simulation object
        max_field (float):              If specified, truncate the color scale of the field plots at that value.
        min_norm_field (float):         If specified, truncate the color scale of the norm field plots below that value.
        max_particle_distance (float):  Show particles that are closer than that distance to the image plane (length
                                        unit, default = inf).
    """
    sys.stdout.write("Compute near field ...\n")
    sys.stdout.flush()
    
    if (not os.path.exists(outputdir)) and (save_plots or save_animations or save_data):
        os.makedirs(outputdir)
    
    if quantities_to_plot is None:
        quantities_to_plot = ['norm(E)']

    vacuum_wavelength = simulation.initial_field.vacuum_wavelength
    if xmin == xmax:
        dim1vec = np.linspace(ymin, ymax, (ymax - ymin)/resolution_step + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin)/resolution_step + 1, endpoint=True)
        yarr, zarr = np.meshgrid(dim1vec, dim2vec)
        xarr = yarr - yarr + xmin
        dim1name = 'y (' + simulation.length_unit + ')'
        dim2name = 'z (' + simulation.length_unit + ')'
    elif ymin == ymax:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin)/resolution_step + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin)/resolution_step + 1, endpoint=True)
        xarr, zarr = np.meshgrid(dim1vec, dim2vec)
        yarr = xarr - xarr + ymin
        dim1name = 'x (' + simulation.length_unit + ')'
        dim2name = 'z (' + simulation.length_unit + ')'
    else:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin)/resolution_step + 1, endpoint=True)
        dim2vec = np.linspace(ymin, ymax, (ymax - ymin)/resolution_step + 1, endpoint=True)
        xarr, yarr = np.meshgrid(dim1vec, dim2vec)
        zarr = xarr - xarr + zmin
        dim1name = 'x (' + simulation.length_unit + ')'
        dim2name = 'y (' + simulation.length_unit + ')'
        
    scat_fld_exp = sf.scattered_field_piecewise_expansion(vacuum_wavelength, simulation.particle_list, 
                                                          simulation.layer_system, k_parallel, azimuthal_angles)
    sys.stdout.write("Evaluate fields ...\n")
    sys.stdout.flush()
    e_x_scat_raw, e_y_scat_raw, e_z_scat_raw = scat_fld_exp.electric_field(xarr, yarr, zarr) 
    
    e_x_init_raw, e_y_init_raw, e_z_init_raw = simulation.initial_field.electric_field(xarr, yarr, zarr,
                                                                                       simulation.layer_system)
    if interpolate_step is None:
        e_x_scat, e_y_scat, e_z_scat = e_x_scat_raw, e_y_scat_raw, e_z_scat_raw
        e_x_init, e_y_init, e_z_init = e_x_init_raw, e_y_init_raw, e_z_init_raw
        dim1vecfine = dim1vec
        dim2vecfine = dim2vec
        interpolate_step = resolution_step
    else:
        sys.stdout.write("Evaluate interpolation ...\n")
        sys.stdout.flush()

        dim1vecfine = np.linspace(dim1vec[0], dim1vec[-1], (dim1vec[-1] - dim1vec[0])/interpolate_step + 1, endpoint=True)
        dim2vecfine = np.linspace(dim2vec[0], dim2vec[-1], (dim2vec[-1] - dim2vec[0])/interpolate_step + 1, endpoint=True)

        real_ex_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_scat_raw.real, 
                                                              kx=interpolation_order, ky=interpolation_order)
        imag_ex_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_scat_raw.imag, 
                                                              kx=interpolation_order, ky=interpolation_order)
        e_x_scat = (real_ex_scat_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ex_scat_interpolant(dim2vecfine, dim1vecfine))

        real_ey_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_scat_raw.real, 
                                                              kx=interpolation_order, ky=interpolation_order)
        imag_ey_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_scat_raw.imag, 
                                                              kx=interpolation_order, ky=interpolation_order)
        e_y_scat = (real_ey_scat_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ey_scat_interpolant(dim2vecfine, dim1vecfine))

        real_ez_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_scat_raw.real, 
                                                              kx=interpolation_order, ky=interpolation_order)
        imag_ez_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_scat_raw.imag, 
                                                              kx=interpolation_order, ky=interpolation_order)
        e_z_scat = (real_ez_scat_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ez_scat_interpolant(dim2vecfine, dim1vecfine))

        real_ex_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_init_raw.real, 
                                                              kx=interpolation_order, ky=interpolation_order)
        imag_ex_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_init_raw.imag, 
                                                              kx=interpolation_order, ky=interpolation_order)
        e_x_init = (real_ex_init_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ex_init_interpolant(dim2vecfine, dim1vecfine))

        real_ey_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_init_raw.real, 
                                                              kx=interpolation_order, ky=interpolation_order)
        imag_ey_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_init_raw.imag, 
                                                              kx=interpolation_order, ky=interpolation_order)
        e_y_init = (real_ey_init_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ey_init_interpolant(dim2vecfine, dim1vecfine))

        real_ez_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_init_raw.real, 
                                                              kx=interpolation_order, ky=interpolation_order)
        imag_ez_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_init_raw.imag, 
                                                              kx=interpolation_order, ky=interpolation_order)
        e_z_init = (real_ez_init_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ez_init_interpolant(dim2vecfine, dim1vecfine))

    if max_field is None:
        vmin = None
        vmax = None
    else:
        vmin = -max_field
        vmax = max_field
        
    sys.stdout.write("Generate final plots ...\n")
    sys.stdout.flush()
    
    for jq, quantity in enumerate(quantities_to_plot):

        filename = 'E'

        fig = plt.figure()
        if 'scat' in quantity:
            e_x, e_y, e_z = e_x_scat, e_y_scat, e_z_scat
            field_type_string = 'scattered electric field'
            filename = filename + '_scat'
        elif 'init' in quantity:
            e_x, e_y, e_z = e_x_init, e_y_init, e_z_init
            field_type_string = 'initial electric field'
            filename = filename + '_init'
        else:
            e_x, e_y, e_z = e_x_scat + e_x_init, e_y_scat + e_y_init, e_z_scat + e_z_init
            field_type_string = 'total electric field'

        if 'norm' in quantity:
            if min_norm_field is None:
                vmin_norm = 0
            else:
                vmin_norm = min_norm_field
            e = np.sqrt(abs(e_x)**2 + abs(e_y)**2 + abs(e_z)**2)
            filename = 'norm_' + filename
            # plt.pcolormesh(dim1vecfine, dim2vecfine, np.sqrt(abs(e_x)**2 + abs(e_y)**2 + abs(e_z)**2), vmin=0,
            #                vmax=vmax, cmap='inferno')
            step2 = interpolate_step/2
            plt.imshow(e, vmin=vmin_norm, vmax=vmax,
                       cmap='inferno',
                       #cmap='jet',
                       extent=[dim1vecfine.min()-step2, dim1vecfine.max()+step2,
                               dim2vecfine.min()-step2, dim2vecfine.max()+step2],
                               interpolation="quadric",
                               #interpolation="none",
                       origin='lower')

            plt_title = 'norm of ' + field_type_string
            plt.title(plt_title)
        else:
            if '_x' in quantity:
                e = e_x
                filename = filename + '_x'
                plt_title = 'x-component of ' + field_type_string
            elif '_y' in quantity:
                e = e_y
                filename = filename + '_y'
                plt_title = 'y-component of ' + field_type_string
            elif '_z' in quantity:
                e = e_z
                filename = filename + '_z'
                plt_title = 'z-component of ' + field_type_string
            # plt.pcolormesh(dim1vecfine, dim2vecfine, e.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
            step2 = interpolate_step/2
            plt.imshow(e.real, vmin=vmin, vmax=vmax, cmap='RdYlBu',
                       extent=[dim1vecfine.min()-step2, dim1vecfine.max()+step2,
                               dim2vecfine.min()-step2, dim2vecfine.max()+step2],
                               #interpolation="quadric",
                               interpolation="none",
                       origin='lower')
            plt.title(plt_title)

        plt.colorbar()
        plt.xlabel(dim1name)
        plt.ylabel(dim2name)

        if not zmin == zmax:
            plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)

        plot_particles(xmin, xmax, ymin, ymax, zmin, zmax, simulation.particle_list, max_particle_distance)

        plt.gca().set_aspect("equal")

        export_filename = outputdir + '/' + filename
        if save_plots:
            plt.savefig(export_filename + '.png')
            #plt.savefig(export_filename + '.pdf')
        if save_animations:
            if not 'norm' in quantity:
                tempdir = tempfile.mkdtemp()
                images = []
                for i_t, t in enumerate(np.linspace(0, 1, 20, endpoint=False)):
                    tempfig = plt.figure()
                    e_t = e * np.exp(-1j * t * 2 * np.pi)
                    # plt.pcolormesh(dim1vecfine, dim2vecfine, e_t.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
                    step2 = interpolate_step/2
                    plt.imshow(e_t.real, vmin=vmin, vmax=vmax, cmap='RdYlBu',
                               extent=[dim1vecfine.min()-step2, dim1vecfine.max()+step2,
                               dim2vecfine.min()-step2, dim2vecfine.max()+step2],
                               #interpolation="quadric",
                               interpolation="none",
                            origin='lower')
                    plt.title(plt_title)
                    plt.colorbar()
                    plt.xlabel(dim1name)
                    plt.ylabel(dim2name)
                    if not zmin == zmax:
                        plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)
                    plot_particles(xmin, xmax, ymin, ymax, zmin, zmax, simulation.particle_list, max_particle_distance)
                    plt.gca().set_aspect("equal")
                    tempfig_filename = tempdir + '/temp_' + str(i_t) + '.png'
                    plt.savefig(tempfig_filename)
                    plt.close(tempfig)
                    images.append(imageio.imread(tempfig_filename))
                imageio.mimsave(export_filename + '.gif', images, duration=0.1)
                shutil.rmtree(tempdir)
        if show_plots:
            plt.draw()
        else:
            plt.close(fig)

    if save_data:
        filename = outputdir + '/spatial_coordinates_along_first_dimension.dat'
        header = dim1name
        np.savetxt(filename, dim1vec, header=header)

        filename = outputdir + '/spatial_coordinates_along_second_dimension.dat'
        header = dim2name
        np.savetxt(filename, dim2vec, header=header)

        filename = outputdir + '/real_e_init_x.dat'
        header = 'Real part of x-component of initial electric field.'
        np.savetxt(filename, e_x_init_raw.real, header=header)

        filename = outputdir + '/imag_e_init_x.dat'
        header = 'Imaginary part of x-component of initial electric field.'
        np.savetxt(filename, e_x_init_raw.imag, header=header)

        filename = outputdir + '/real_e_init_y.dat'
        header = 'Real part of y-component of initial electric field.'
        np.savetxt(filename, e_y_init_raw.real, header=header)

        filename = outputdir + '/imag_e_init_y.dat'
        header = 'Imaginary part of y-component of initial electric field.'
        np.savetxt(filename, e_y_init_raw.imag, header=header)

        filename = outputdir + '/real_e_init_z.dat'
        header = 'Real part of z-component of initial electric field.'
        np.savetxt(filename, e_z_init_raw.real, header=header)

        filename = outputdir + '/imag_e_init_z.dat'
        header = 'Imaginary part of z-component of initial electric field.'
        np.savetxt(filename, e_z_init_raw.imag, header=header)

        filename = outputdir + '/real_e_scat_x.dat'
        header = 'Real part of x-component of scattered electric field.'
        np.savetxt(filename, e_x_scat_raw.real, header=header)

        filename = outputdir + '/imag_e_scat_x.dat'
        header = 'Imaginary part of x-component of scattered electric field.'
        np.savetxt(filename, e_x_scat_raw.imag, header=header)

        filename = outputdir + '/real_e_scat_y.dat'
        header = 'Real part of y-component of scattered electric field.'
        np.savetxt(filename, e_y_scat_raw.real, header=header)

        filename = outputdir + '/imag_e_scat_y.dat'
        header = 'Imaginary part of y-component of scattered electric field.'
        np.savetxt(filename, e_y_scat_raw.imag, header=header)

        filename = outputdir + '/real_e_scat_z.dat'
        header = 'Real part of z-component of scattered electric field.'
        np.savetxt(filename, e_z_scat_raw.real, header=header)

        filename = outputdir + '/imag_e_scat_z.dat'
        header = 'Imaginary part of z-component of scattered electric field.'
        np.savetxt(filename, e_z_scat_raw.imag, header=header)


def show_far_field(far_field, save_plots, show_plots, save_data=False, tag='far_field', outputdir='.', 
                   flip_downward=True, split=True):
    """Display and export the far field.
    
    Args:
        far_field (smuthi.field_expansion.FarField):    far field object to show and export
        save_plots (bool):                              save images if true
        show_plots (bool):                              display plots if true
        save_data (bool):                               export data in ascii format if true
        tag (str):                                      name to attribute files
        outputdir (str):                                path to the directory where data to be stored
        flip_downward (bool):                           represent downward directions as 0-90 deg instead of 90-180
                                                        if true
        split (bool):                                   show two different plots for upward and downward directions 
                                                        if true
    """
    
    if split and any(far_field.polar_angles < np.pi/2) and any(far_field.polar_angles > np.pi/2):
        show_far_field(far_field.top(), save_plots, show_plots, save_data, tag+'_top', outputdir, True, False)
        show_far_field(far_field.bottom(), save_plots, show_plots, save_data, tag+'_bottom', outputdir, True, False)
        return
    
    if (not os.path.exists(outputdir)) and (save_plots or save_data):
        os.makedirs(outputdir)
    
    if save_data:
        far_field.export(output_directory=outputdir, tag=tag)
    
    alpha_grid = far_field.alpha_grid()
    beta_grid = far_field.beta_grid()

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    if flip_downward and all(far_field.polar_angles >= np.pi / 2):
        pcm = ax.pcolormesh(alpha_grid, 180 - beta_grid, (far_field.signal[0, :, :] + far_field.signal[1, :, :]),
                            cmap='inferno')
    else:
        pcm = ax.pcolormesh(alpha_grid, beta_grid, (far_field.signal[0, :, :] + far_field.signal[1, :, :]),
                            cmap='inferno')
    plt.colorbar(pcm, ax=ax)
    plt.title(tag)
    if save_plots:
        plt.savefig(outputdir + '/' + tag + '.png')
    if show_plots:
        plt.draw()
    else:
        plt.close(fig)
    fig = plt.figure()
    if flip_downward and all(far_field.polar_angles >= np.pi/2):
        plt.plot(180 - far_field.polar_angles * 180 / np.pi, np.sum(far_field.azimuthal_integral(), axis=0) * np.pi / 180)
    else:
        plt.plot(far_field.polar_angles * 180 / np.pi, np.sum(far_field.azimuthal_integral(), axis=0) * np.pi / 180)

    plt.xlabel('polar angle (degree)')
    if far_field.signal_type == 'differential cross section':
        plt.ylabel('d_CS/d_beta')
    elif far_field.signal_type == 'intensity':
        plt.ylabel('d_P/d_beta')
    plt.grid(True)
    plt.title(tag)
    if save_plots:
        plt.savefig(outputdir + '/' + tag + '_polar.png')
    if show_plots:
        plt.draw()
    else:
        plt.close(fig)
