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
    for particle in particle_list:
        pos = particle.position
        if xmin == xmax:
            if abs(xmin - pos[0]) < max_particle_distance:
                if type(particle).__name__ == 'Sphere':
                    plt.gca().add_patch(Circle((pos[1], pos[2]), particle['radius'], facecolor='w', edgecolor='k'))
                elif type(particle).__name__ == 'Spheroid':
                    if not particle.euler_angles == [0, 0, 0]:
                        raise ValueError('rotated particles currently not supported')
                    plt.gca().add_patch(Ellipse(xy=(pos[1], pos[2]), width=2 * particle['semi axis a'],
                                                height=2 * particle.semi_axis_c, facecolor='w', edgecolor='k'))
                    circumscribing_radius = max([particle.semi_axis_a, particle.semi_axis_c])
                    plt.gca().add_patch(Circle((pos[1], pos[2]), circumscribing_radius, linestyle='dashed',
                                               facecolor='none', edgecolor='k'))
                elif type(particle).__name__ == 'FiniteCylinder':
                    cylinder_radius = particle.cylinder_radius
                    cylinder_height = particle.cylinder_height
                    plt.gca().add_patch(Rectangle((pos[1] - cylinder_radius, pos[2] - cylinder_height / 2),
                                                  2 * cylinder_radius, cylinder_height, facecolor='w', edgecolor='k'))
                    circumscribing_radius = np.sqrt(particle.cylinder_height ** 2 / 4 + particle.cylinder_radius ** 2)
                    plt.gca().add_patch(Circle((pos[1], pos[2]), circumscribing_radius, linestyle='dashed',
                                               facecolor='none', edgecolor='k'))

        elif ymin == ymax:
            if abs(ymin - pos[1]) < max_particle_distance:
                if type(particle).__name__ == 'Sphere':
                    plt.gca().add_patch(Circle((pos[0], pos[2]), particle.radius, facecolor='w', edgecolor='k'))
                elif type(particle).__name__ == 'Spheroid':
                    if not particle.euler_angles == [0, 0, 0]:
                        raise ValueError('rotated particles currently not supported')
                    plt.gca().add_patch(Ellipse(xy=(pos[0], pos[2]), width=2 * particle.semi_axis_a,
                                                height=2 * particle.semi_axis_c, facecolor='w', edgecolor='k'))
                    circumscribing_radius = max([particle.semi_axis_a, particle.semi_axis_c])
                    plt.gca().add_patch(Circle((pos[0], pos[2]), circumscribing_radius, linestyle='dashed',
                                               facecolor='none', edgecolor='k'))
                elif type(particle).__name__ == 'FiniteCylinder':
                    cylinder_radius = particle.cylinder_radius
                    cylinder_height = particle.cylinder_height
                    plt.gca().add_patch(Rectangle((pos[0] - cylinder_radius, pos[2] - cylinder_height / 2),
                                                  2 * cylinder_radius, cylinder_height, facecolor='w', edgecolor='k'))
                    circumscribing_radius = np.sqrt(cylinder_height ** 2 / 4 + cylinder_radius ** 2)
                    plt.gca().add_patch(Circle((pos[0], pos[2]), circumscribing_radius, linestyle='dashed',
                                               facecolor='none', edgecolor='k'))

        elif zmin == zmax:
            if abs(zmin - pos[2]) < max_particle_distance:
                if type(particle).__name__ == 'Sphere':
                    plt.gca().add_patch(Circle((pos[0], pos[1]), particle.radius, facecolor='w', edgecolor='k'))
                elif type(particle).__name__ == 'Spheroid':
                    if not particle.euler_angles == [0, 0, 0]:
                        raise ValueError('rotated particles currently not supported')
                    plt.gca().add_patch(Circle((pos[0], pos[1]), particle.semi_axis_a, facecolor='w', edgecolor='k'))
                elif type(particle).__name__ == 'FiniteCylinder':
                    cylinder_radius = particle.cylinder_radius
                    plt.gca().add_patch(Circle((pos[0], pos[1]), cylinder_radius, facecolor='w', edgecolor='k'))


def show_near_field(quantities_to_plot=None, save_plots=False, show_plots=True, save_animations=False, save_data=False,
                    outputdir='.', xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0, resolution=25, interpolate=None,
                    k_parallel=None, azimuthal_angles=None, simulation=None, max_field=None,
                    max_particle_distance=float('inf')):
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
        resolution (float):     Compute the field with that spatial resolution (length unit)
        interpolate (float):    Use spline interpolation with that resolution to plot a smooth field (length unit)
        k_parallel (array):         1-D Numpy array of in-plane wavenumbers for the plane wave expansion
        azimuthal_angles (array):   1-D Numpy array of azimuthal angles for the plane wave expansion
        simulation (smuthi.simulation.Simulation):  Simulation object
        max_field (float):              If specified, truncate the color scale of the field plots at that value.
        max_particle_distance (float):  Show particles that are closer than that distance to the image plane (length
                                        unit, default = inf).
    """
    if quantities_to_plot is None:
        quantities_to_plot = ['norm(E)']

    vacuum_wavelength = simulation.initial_field.vacuum_wavelength
    if xmin == xmax:
        dim1vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        yarr, zarr = np.meshgrid(dim1vec, dim2vec)
        xarr = yarr - yarr + xmin
        dim1name = 'y (' + simulation.length_unit + ')'
        dim2name = 'z (' + simulation.length_unit + ')'
    elif ymin == ymax:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        xarr, zarr = np.meshgrid(dim1vec, dim2vec)
        yarr = xarr - xarr + ymin
        dim1name = 'x (' + simulation.length_unit + ')'
        dim2name = 'z (' + simulation.length_unit + ')'
    else:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        xarr, yarr = np.meshgrid(dim1vec, dim2vec)
        zarr = xarr - xarr + zmin
        dim1name = 'x (' + simulation.length_unit + ')'
        dim2name = 'y (' + simulation.length_unit + ')'
        
    scat_fld_exp = sf.scattered_field_piecewise_expansion(k_parallel, azimuthal_angles, vacuum_wavelength,
                                                          simulation.particle_list, simulation.layer_system)
    e_x_scat_raw, e_y_scat_raw, e_z_scat_raw = scat_fld_exp.electric_field(xarr, yarr, zarr) 
    
    e_x_init_raw, e_y_init_raw, e_z_init_raw = simulation.initial_field.electric_field(xarr, yarr, zarr,
                                                                                       simulation.layer_system)
    if interpolate is None:
        e_x_scat, e_y_scat, e_z_scat = e_x_scat_raw, e_y_scat_raw, e_z_scat_raw
        e_x_init, e_y_init, e_z_init = e_x_init_raw, e_y_init_raw, e_z_init_raw
        dim1vecfine = dim1vec
        dim2vecfine = dim2vec
    else:
        dim1vecfine = np.linspace(dim1vec[0], dim1vec[-1], (dim1vec[-1] - dim1vec[0]) / interpolate + 1, endpoint=True)
        dim2vecfine = np.linspace(dim2vec[0], dim2vec[-1], (dim2vec[-1] - dim2vec[0]) / interpolate + 1, endpoint=True)

        real_ex_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_scat_raw.real)
        imag_ex_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_scat_raw.imag)
        e_x_scat = (real_ex_scat_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ex_scat_interpolant(dim2vecfine, dim1vecfine))

        real_ey_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_scat_raw.real)
        imag_ey_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_scat_raw.imag)
        e_y_scat = (real_ey_scat_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ey_scat_interpolant(dim2vecfine, dim1vecfine))

        real_ez_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_scat_raw.real)
        imag_ez_scat_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_scat_raw.imag)
        e_z_scat = (real_ez_scat_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ez_scat_interpolant(dim2vecfine, dim1vecfine))

        real_ex_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_init_raw.real)
        imag_ex_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_init_raw.imag)
        e_x_init = (real_ex_init_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ex_init_interpolant(dim2vecfine, dim1vecfine))

        real_ey_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_init_raw.real)
        imag_ey_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_init_raw.imag)
        e_y_init = (real_ey_init_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ey_init_interpolant(dim2vecfine, dim1vecfine))

        real_ez_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_init_raw.real)
        imag_ez_init_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_init_raw.imag)
        e_z_init = (real_ez_init_interpolant(dim2vecfine, dim1vecfine)
                    + 1j * imag_ez_init_interpolant(dim2vecfine, dim1vecfine))

    if max_field is None:
        vmin = None
        vmax = None
    else:
        vmin = -max_field
        vmax = max_field

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
            e = np.sqrt(abs(e_x)**2 + abs(e_y)**2 + abs(e_z)**2)
            filename = 'norm_' + filename
            plt.pcolormesh(dim1vecfine, dim2vecfine, np.sqrt(abs(e_x)**2 + abs(e_y)**2 + abs(e_z)**2), vmin=0,
                           vmax=vmax, cmap='inferno')
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
            plt.pcolormesh(dim1vecfine, dim2vecfine, e.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
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
        if save_animations:
            if not 'norm' in quantity:
                tempdir = tempfile.mkdtemp()
                images = []
                for i_t, t in enumerate(np.linspace(0, 1, 20, endpoint=False)):
                    tempfig = plt.figure()
                    e_t = e * np.exp(-1j * t * 2 * np.pi)
                    plt.pcolormesh(dim1vecfine, dim2vecfine, e_t.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
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


def scattered_electric_field(x, y, z, k_parallel, azimuthal_angles, vacuum_wavelength, particle_list, layer_system):
    """Complex electric scattered near field.

    Args:
        x (numpy array):    x-coordinates of points in space where to evaluate field.
        y (numpy array):    y-coordinates of points in space where to evaluate field.
        z (numpy array):    z-coordinates of points in space where to evaluate field.
        k_parallel (1D numpy array):        In plane wavenumbers for the plane wave expansion
        azimuthal_angles (1D numpy array):  Azimuthal angles for the plane wave expansion
        vacuum_wavelength (float):          Vacuum wavelength
        particle_list (list):               List of smuthi.particle.Particle objects
        layer_system (smuthi.layers.LayerSystem):   Stratified medium
    Returns:
        Tuple (E_x, E_y, E_z) of complex electric field values as numpy arrays. The shapes are the same as x.shape
    """

    old_shape = x.shape
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    electric_field_x = np.zeros(x.shape, dtype=complex)
    electric_field_y = np.zeros(x.shape, dtype=complex)
    electric_field_z = np.zeros(x.shape, dtype=complex)

    layer_numbers = []
    for zi in z:
        layer_numbers.append(layer_system.layer_number(zi))

    for i in range(layer_system.number_of_layers()):
        layer_indices = [ii for ii, laynum in enumerate(layer_numbers) if laynum == i]
        if layer_indices:

            # layer mediated scattered field ---------------------------------------------------------------------------
            k = coord.angular_frequency(vacuum_wavelength) * layer_system.refractive_indices[i]
            ref = [0, 0, layer_system.reference_z(i)]
            vb = (layer_system.lower_zlimit(i), layer_system.upper_zlimit(i))
            pwe_up = fldex.PlaneWaveExpansion(k=k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles,
                                              kind='upgoing', reference_point=ref, lower_z=vb[0], upper_z=vb[1])
            pwe_down = fldex.PlaneWaveExpansion(k=k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles,
                                                kind='downgoing', reference_point=ref, lower_z=vb[0], upper_z=vb[1])
            for particle in particle_list:
                add_up, add_down = fldex.swe_to_pwe_conversion(particle.scattered_field, k_parallel, azimuthal_angles,
                                                               layer_system, i, True)
                pwe_up = pwe_up + add_up
                pwe_down = pwe_down + add_down
            ex_up, ey_up, ez_up = pwe_up.electric_field(x[layer_indices], y[layer_indices], z[layer_indices])
            ex_down, ey_down, ez_down = pwe_down.electric_field(x[layer_indices], y[layer_indices], z[layer_indices])
            electric_field_x[layer_indices] = ex_up + ex_down
            electric_field_y[layer_indices] = ey_up + ey_down
            electric_field_z[layer_indices] = ez_up + ez_down

            # direct field ---------------------------------------------------------------------------------------------
            for particle in particle_list:
                if layer_system.layer_number(particle.position[2]) == i:
                    ex, ey, ez = particle.scattered_field.electric_field(x[layer_indices], y[layer_indices],
                                                                         z[layer_indices])
                    electric_field_x[layer_indices] += ex
                    electric_field_y[layer_indices] += ey
                    electric_field_z[layer_indices] += ez
                    # todo:check if swe valid, fill in NaN or something otherwise

    return electric_field_x.reshape(old_shape), electric_field_y.reshape(old_shape), electric_field_z.reshape(old_shape)
