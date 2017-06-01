# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as interp
import smuthi.coordinates as coord
import smuthi.plane_wave_pattern as pwp
import smuthi.vector_wave_functions as vwf
import smuthi.index_conversion as idx
import smuthi.layers as lay
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
import tempfile
import shutil
import imageio


def plot_layer_interfaces(dim1min, dim1max, layer_system):
    """Add lines to plot to display layer system interfaces

    input:
    dim1min:        From what x-value plot line
    dim1max:        To what x-value plot line
    layer_system:   Object of class smuthi.layers.LayerSystem
    """
    for il in range(1, layer_system.number_of_layers()):
        plt.plot([dim1min, dim1max], [layer_system.reference_z(il), layer_system.reference_z(il)], 'g')


def plot_particles(xmin, xmax, ymin, ymax, zmin, zmax, particle_collection, max_particle_distance):
    """Add circles, ellipses and rectangles to plot to display spheres, spheroids and cylinders.

    xmin:                   Minimal x-value of plot
    xmax:                   Maximal x-value of plot
    ymin:                   Minimal y-value of plot
    ymax:                   Maximal y-value of plot
    zmin:                   Minimal z-value of plot
    zmax:                   Maximal z-value of plot
    particle_collection:    Object of class smuthi.particles.ParticleCollection
    max_particle_distance:  Plot only particles that ar not further away from image plane
    """
    for particle in particle_collection.particles:
        pos = particle['position']
        if xmin == xmax:
            if abs(xmin - pos[0]) < max_particle_distance:
                if particle['shape'] == 'sphere':
                    plt.gca().add_patch(Circle((pos[1], pos[2]), particle['radius'], facecolor='w', edgecolor='k'))
                elif particle['shape'] == 'spheroid':
                    if not particle['euler angles'] == [0, 0, 0]:
                        raise ValueError('rotated particles currently not supported')
                    plt.gca().add_patch(Ellipse(xy=(pos[1], pos[2]), width=2 * particle['semi axis a'],
                                                height=2 * particle['semi axis c'], facecolor='w', edgecolor='k'))
                    circumscribing_radius = max([particle['semi axis a'], particle['semi axis c']])
                    plt.gca().add_patch(Circle((pos[1], pos[2]), circumscribing_radius, linestyle='dashed',
                                               facecolor='none', edgecolor='k'))
                elif particle['shape'] == 'finite cylinder':
                    cylinder_radius = particle['cylinder radius']
                    cylinder_height = particle['cylinder height']
                    plt.gca().add_patch(Rectangle((pos[1] - cylinder_radius, pos[2] - cylinder_height / 2),
                                                  2 * cylinder_radius, cylinder_height, facecolor='w',
                                                  edgecolor='k'))
                    circumscribing_radius = np.sqrt(particle['cylinder height'] ** 2 / 4
                                                    + particle['cylinder radius'] ** 2)
                    plt.gca().add_patch(Circle((pos[1], pos[2]), circumscribing_radius, linestyle='dashed',
                                               facecolor='none', edgecolor='k'))

        elif ymin == ymax:
            if abs(ymin - pos[1]) < max_particle_distance:
                if particle['shape'] == 'sphere':
                    plt.gca().add_patch(Circle((pos[0], pos[2]), particle['radius'], facecolor='w', edgecolor='k'))
                elif particle['shape'] == 'spheroid':
                    if not particle['euler angles'] == [0, 0, 0]:
                        raise ValueError('rotated particles currently not supported')
                    plt.gca().add_patch(Ellipse(xy=(pos[0], pos[2]), width=2 * particle['semi axis a'],
                                                height=2 * particle['semi axis c'], facecolor='w', edgecolor='k'))
                    circumscribing_radius = max([particle['semi axis a'], particle['semi axis c']])
                    plt.gca().add_patch(Circle((pos[0], pos[2]), circumscribing_radius, linestyle='dashed',
                                               facecolor='none', edgecolor='k'))
                elif particle['shape'] == 'finite cylinder':
                    cylinder_radius = particle['cylinder radius']
                    cylinder_height = particle['cylinder height']
                    plt.gca().add_patch(Rectangle((pos[0] - cylinder_radius, pos[2] - cylinder_height / 2),
                                                  2 * cylinder_radius, cylinder_height, facecolor='w',
                                                  edgecolor='k'))
                    circumscribing_radius = np.sqrt(particle['cylinder height'] ** 2 / 4
                                                    + particle['cylinder radius'] ** 2)
                    plt.gca().add_patch(Circle((pos[0], pos[2]), circumscribing_radius, linestyle='dashed',
                                               facecolor='none', edgecolor='k'))

        elif zmin == zmax:
            if abs(zmin - pos[2]) < max_particle_distance:
                if particle['shape'] == 'sphere':
                    plt.gca().add_patch(Circle((pos[0], pos[1]), particle['radius'], facecolor='w', edgecolor='k'))
                elif particle['shape'] == 'spheroid':
                    if not particle['euler angles'] == [0, 0, 0]:
                        raise ValueError('rotated particles currently not supported')
                    plt.gca().add_patch(Circle((pos[0], pos[1]), particle['semi axis a'], facecolor='w',
                                               edgecolor='k'))
                elif particle['shape'] == 'finite cylinder':
                    cylinder_radius = particle['cylinder radius']
                    plt.gca().add_patch(Circle((pos[0], pos[1]), cylinder_radius, facecolor='w', edgecolor='k'))


def show_near_field(quantities_to_plot, filenames=None, xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0, resolution=25, interpolate=None,
                    n_effective=None, azimuthal_angles=None, simulation=None, layerresponse_precision=None,
                    max_field=None, max_particle_distance=float('inf')):
    """Plot the electric near field along a plane. To plot along the xy-plane, specify zmin=zmax and so on.

    input:
    quantities_to_plot:         List of strings that specify what to plot. Select from 'E_x', 'E_y', 'E_z', 'norm(E)'

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

    filenames:                  List of strings to specify the path where to store the near field images. Filenames can
                                end on .png or on .gif (to create animated plots)
    xmin:                       Plot from that x (length unit)
    xmax:                       Plot up to that x (length unit)
    ymin:                       Plot from that y (length unit)
    ymax:                       Plot up to that y (length unit)
    zmin:                       Plot from that z (length unit)
    zmax:                       Plot up to that z (length unit)
    resolution:                 Compute the field with that spatial resolution (length unit)
    interpolate:                Use spline interpolation with that resolution to plot a smooth field (length unit)
    n_effective:                1-D Numpy array of effective refractive index for the plane wave expansion
    azimuthal_angles:           1-D Numpy array of azimuthal angles for the plane wave expansion
    simulation:                 smuthi.simulation.Simulation object
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    max_field:                  If specified, truncate the color scale of the field plots at that value.
    max_particle_distance       Show particles that are closer than that distance to the image plane
                                (length unit, default = inf).
    """
    vacuum_wavelength = simulation.initial_field_collection.vacuum_wavelength
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

    e_x_scat_raw, e_y_scat_raw, e_z_scat_raw = scattered_electric_field(
        xarr, yarr, zarr, n_effective, azimuthal_angles, vacuum_wavelength, simulation.particle_collection,
        simulation.linear_system, simulation.layer_system, layerresponse_precision)

    e_x_init_raw, e_y_init_raw, e_z_init_raw = initial_electric_field(
        xarr, yarr, zarr, simulation.initial_field_collection, simulation.layer_system, layerresponse_precision)

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

        plt.figure()
        if 'scat' in quantity:
            e_x, e_y, e_z = e_x_scat, e_y_scat, e_z_scat
            field_type_string = 'scattered electric field'
        elif 'init' in quantity:
            e_x, e_y, e_z = e_x_init, e_y_init, e_z_init
            field_type_string = 'initial electric field'
        else:
            e_x, e_y, e_z = e_x_scat + e_x_init, e_y_scat + e_y_init, e_z_scat + e_z_init
            field_type_string = 'total electric field'

        if 'norm' in quantity:
            e = np.sqrt(abs(e_x)**2 + abs(e_y)**2 + abs(e_z)**2)
            plt.pcolormesh(dim1vecfine, dim2vecfine, np.sqrt(abs(e_x)**2 + abs(e_y)**2 + abs(e_z)**2), vmin=0,
                           vmax=vmax, cmap='inferno')
            plt_title = 'norm of ' + field_type_string
            plt.title(plt_title)
        else:
            if '_x' in quantity:
                e = e_x
                plt_title = 'x-component of ' + field_type_string
            elif '_y' in quantity:
                e = e_y
                plt_title = 'y-component of ' + field_type_string
            elif '_z' in quantity:
                e = e_z
                plt_title = 'z-component of ' + field_type_string
            plt.pcolormesh(dim1vecfine, dim2vecfine, e.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
            plt.title(plt_title)

        plt.colorbar()
        plt.xlabel(dim1name)
        plt.ylabel(dim2name)

        if not zmin == zmax:
            plot_layer_interfaces(dim1vec[0], dim1vec[-1], simulation.layer_system)

        plot_particles(xmin, xmax, ymin, ymax, zmin, zmax, simulation.particle_collection, max_particle_distance)

        plt.gca().set_aspect("equal")

        try:
            export_filename = filenames[jq]
            if export_filename[-3:] == 'png':
                plt.savefig(export_filename)
            elif export_filename[-3:] == 'gif':
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
                    plot_particles(xmin, xmax, ymin, ymax, zmin, zmax, simulation.particle_collection,
                                   max_particle_distance)
                    plt.gca().set_aspect("equal")
                    tempfig_filename = tempdir + '/temp_' + str(i_t) + '.png'
                    plt.savefig(tempfig_filename)
                    plt.close(tempfig)
                    images.append(imageio.imread(tempfig_filename))
                imageio.mimsave(export_filename, images, duration=0.1)
                shutil.rmtree(tempdir)
        except:
            pass

    plt.draw()


def scattered_electric_field(x, y, z, n_effective=None, azimuthal_angles=None, vacuum_wavelength=None,
                             particle_collection=None, linear_system=None, layer_system=None,
                             layerresponse_precision=None):
    """Complex electric scttered near field.
    Return the x, y and z component of the scattered electric field.

    NOT TESTED

    input:
    x:                          Numpy array of x-coordinates of points in space where to evaluate field.
    y:                          Numpy array of y-coordinates of points in space where to evaluate field.
    z:                          Numpy array of z-coordinates of points in space where to evaluate field.
    n_effective:                1-D Numpy array of effective refractive index for the plane wave expansion
    azimuthal_angles:           1-D Numpy array of azimuthal angles for the plane wave expansion
    vacuum_wavelength:          Vacuum wavelength
    particle_collection:        smuthi.particle.ParticleCollection object
    linear_system:              smuthi.linear_system.LinearSystem object
    layer_system:               smuthi.layers.LayerSystem object
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """

    old_shape = x.shape
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    electric_field_x = np.zeros(x.shape, dtype=complex)
    electric_field_y = np.zeros(x.shape, dtype=complex)
    electric_field_z = np.zeros(x.shape, dtype=complex)

    kp = n_effective * coord.angular_frequency(vacuum_wavelength)

    layer_numbers = []
    for zi in z:
        layer_numbers.append(layer_system.layer_number(zi))

    particle_positions = particle_collection.particle_positions()
    particle_layer_numbers = []
    for pos in particle_positions:
        particle_layer_numbers.append(layer_system.layer_number(pos[2]))

    for il in range(layer_system.number_of_layers()):
        layer_indices = [i for i, laynum in enumerate(layer_numbers) if laynum == il]
        if layer_indices:

            # indirect field -------------------------------------------------------------------------------------------
            kz = coord.k_z(k_parallel=kp, vacuum_wavelength=vacuum_wavelength,
                           refractive_index=layer_system.refractive_indices[il])
            xil = x[layer_indices]
            yil = y[layer_indices]
            zil = z[layer_indices] - layer_system.reference_z(il)
            pwp_rs = pwp.plane_wave_pattern_rs(n_effective=n_effective, azimuthal_angles=azimuthal_angles,
                                              vacuum_wavelength=vacuum_wavelength,
                                              particle_collection=particle_collection, linear_system=linear_system,
                                              layer_system=layer_system, layer_numbers=[il],
                                              layerresponse_precision=layerresponse_precision)[0]

            for pol in range(2):
                integrand_alpha_x = np.zeros((len(azimuthal_angles), len(layer_indices)), dtype=complex)
                integrand_alpha_y = np.zeros((len(azimuthal_angles), len(layer_indices)), dtype=complex)
                integrand_alpha_z = np.zeros((len(azimuthal_angles), len(layer_indices)), dtype=complex)

                for ja, a in enumerate(azimuthal_angles):
                    e_x_pl, e_y_pl, e_z_pl = vwf.plane_vector_wave_function(xil[:, None], yil[:, None], zil[:, None],
                                                                            kp[None, :], a, kz[None, :], pol)
                    e_x_mn, e_y_mn, e_z_mn = vwf.plane_vector_wave_function(xil[:, None], yil[:, None], zil[:, None],
                                                                            kp[None, :], a, -kz[None, :], pol)

                    integrand_k = kp * pwp_rs[pol, 0, :, ja] * e_x_pl + kp * pwp_rs[pol, 1, :, ja] * e_x_mn
                    integrand_k[np.isnan(integrand_k)] = 0
                    integrand_alpha_x[ja, :] = np.trapz(integrand_k, kp)

                    integrand_k = kp * pwp_rs[pol, 0, :, ja] * e_y_pl + kp * pwp_rs[pol, 1, :, ja] * e_y_mn
                    integrand_k[np.isnan(integrand_k)] = 0
                    integrand_alpha_y[ja, :] = np.trapz(integrand_k, kp)

                    integrand_k = kp * pwp_rs[pol, 0, :, ja] * e_z_pl + kp * pwp_rs[pol, 1, :, ja] * e_z_mn
                    integrand_k[np.isnan(integrand_k)] = 0
                    integrand_alpha_z[ja, :] = np.trapz(integrand_k, kp)

                electric_field_x[layer_indices] += np.trapz(integrand_alpha_x, azimuthal_angles, axis=0)
                electric_field_y[layer_indices] += np.trapz(integrand_alpha_y, azimuthal_angles, axis=0)
                electric_field_z[layer_indices] += np.trapz(integrand_alpha_z, azimuthal_angles, axis=0)

            # direct field ---------------------------------------------------------------------------------------------
            k = coord.angular_frequency(vacuum_wavelength) * layer_system.refractive_indices[il]
            for js in range(particle_collection.particle_number()):
                if particle_layer_numbers[js] == il:
                    for tau in range(2):
                        for m in range(-idx.m_max, idx.m_max + 1):
                            for l in range(max(abs(m), 1), idx.l_max + 1):
                                n = idx.multi_to_single_index(tau, l, m)
                                b = linear_system.scattered_field_coefficients[js, n]
                                xrel = xil - particle_positions[js][0]
                                yrel = yil - particle_positions[js][1]
                                zrel = z[layer_indices] - particle_positions[js][2]
                                e_x_svwf, e_y_svwf, e_z_svwf = vwf.spherical_vector_wave_function(xrel, yrel, zrel, k,
                                                                                                  3, tau, l, m)
                                electric_field_x[layer_indices] += e_x_svwf * b
                                electric_field_y[layer_indices] += e_y_svwf * b
                                electric_field_z[layer_indices] += e_z_svwf * b

    return electric_field_x.reshape(old_shape), electric_field_y.reshape(old_shape), electric_field_z.reshape(old_shape)


def initial_electric_field(x, y, z, initial_field_collection, layer_system, layerresponse_precision=None):
    """Return the x, y and z component of the initial electric field (including layer system response).

    input:
    x:                          Numpy array of x-coordinates of points in space where to evaluate field.
    y:                          Numpy array of y-coordinates of points in space where to evaluate field.
    z:                          Numpy array of z-coordinates of points in space where to evaluate field.
    initial_field_collection    smuthi.initial.InitialFieldCollection object
    layer_system:               smuthi.layers.LayerSystem object
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    electric_field_x = np.zeros(x.shape, dtype=complex)
    electric_field_y = np.zeros(x.shape, dtype=complex)
    electric_field_z = np.zeros(x.shape, dtype=complex)

    vacuum_wavelength = initial_field_collection.vacuum_wavelength
    for field_specs in initial_field_collection.specs_list:
        if field_specs['type'] == 'plane wave':
            e_x, e_y, e_z = plane_wave_electric_field(x, y, z, vacuum_wavelength=vacuum_wavelength,
                                                      polarization=field_specs['polarization'],
                                                      polar_angle=field_specs['polar angle'],
                                                      azimuthal_angle=field_specs['azimuthal angle'],
                                                      amplitude=field_specs['amplitude'],
                                                      reference_point=field_specs['reference point'],
                                                      layer_system=layer_system,
                                                      layerresponse_precision=layerresponse_precision)
            electric_field_x += e_x
            electric_field_y += e_y
            electric_field_z += e_z

    return e_x, e_y, e_z


def plane_wave_electric_field(x, y, z, vacuum_wavelength, polarization, polar_angle, azimuthal_angle, amplitude=1,
                              reference_point=[0, 0, 0], layer_system=None, layerresponse_precision=None):
    """Return the x, y and z component of the electric field of a plane wave (including layer system response).

    input:
    x:                          Numpy array of x-coordinates of points in space where to evaluate field.
    y:                          Numpy array of y-coordinates of points in space where to evaluate field.
    z:                          Numpy array of z-coordinates of points in space where to evaluate field.
    vacuum_wavelength:          Vacuum wavelength.
    polarization:               Polarization (0 for TE, 1 for TM)
    polar_angle:                Polar angle of propagation vector of plane wave (radians)
    azimuthal_angle:            Azimuthal angle of propagation vector of plane wave (radians)
    amplitude:                  Complex amplitude of wave (at reference point) (default=1)
    reference_pont:             List in the format [x, y, z] containing the coordinates of the reference point where the
                                electric field equals the amplitude parameter
    layer_system:               smuthi.layers.LayerSystem object
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    old_shape = x.shape
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    electric_field_x = np.zeros(x.shape, dtype=complex)
    electric_field_y = np.zeros(x.shape, dtype=complex)
    electric_field_z = np.zeros(x.shape, dtype=complex)

    omega = coord.angular_frequency(vacuum_wavelength)

    i_top = layer_system.number_of_layers() - 1
    if polar_angle < np.pi / 2:
        i_P = 0
    else:
        i_P = i_top
    n_P = layer_system.refractive_indices[i_P]
    k_P = n_P * omega

    # complex amplitude of initial wave (including phase factor for reference point)
    kappa_P = np.sin(polar_angle) * k_P
    kx = np.cos(azimuthal_angle) * kappa_P
    ky = np.sin(azimuthal_angle) * kappa_P
    pm_kz_P = k_P * np.cos(polar_angle)
    kvec_P = np.array([kx, ky, pm_kz_P])
    rvec_iP = np.array([0, 0, layer_system.reference_z(i_P)])
    rvec_0 = np.array(reference_point)
    ejkriP = np.exp(1j * np.dot(kvec_P, rvec_iP - rvec_0))
    A_P = amplitude * ejkriP
    if i_P == 0:
        gexc = np.array([A_P, 0])
    else:
        gexc = np.array([0, A_P])

    # which field point is in which layer?
    layer_numbers = []
    for zi in z:
        layer_numbers.append(layer_system.layer_number(zi))

    for il in range(layer_system.number_of_layers()):
        layer_indices = [i for i, laynum in enumerate(layer_numbers) if laynum == il]
        if layer_indices:
            L = lay.layersystem_response_matrix(polarization, layer_system.thicknesses, layer_system.refractive_indices,
                                                kappa_P, omega, i_P, il, layerresponse_precision)
            gil = np.dot(L, gexc)
            kz = coord.k_z(k_parallel=kappa_P, vacuum_wavelength=vacuum_wavelength,
                           refractive_index=layer_system.refractive_indices[il])
            xil = x[layer_indices]
            yil = y[layer_indices]
            zil = z[layer_indices] - layer_system.reference_z(il)
            e_x_pl, e_y_pl, e_z_pl = vwf.plane_vector_wave_function(xil, yil, zil, kappa_P, azimuthal_angle, kz,
                                                                    polarization)
            e_x_mn, e_y_mn, e_z_mn = vwf.plane_vector_wave_function(xil, yil, zil, kappa_P, azimuthal_angle, -kz,
                                                                    polarization)
            electric_field_x[layer_indices] += e_x_pl * gil[0] + e_x_mn * gil[1]
            electric_field_y[layer_indices] += e_y_pl * gil[0] + e_y_mn * gil[1]
            electric_field_z[layer_indices] += e_z_pl * gil[0] + e_z_mn * gil[1]

            if il == i_P:  # add direct contribution
                electric_field_x[layer_indices] += e_x_pl * gexc[0] + e_x_mn * gexc[1]
                electric_field_y[layer_indices] += e_y_pl * gexc[0] + e_y_mn * gexc[1]
                electric_field_z[layer_indices] += e_z_pl * gexc[0] + e_z_mn * gexc[1]

    return electric_field_x.reshape(old_shape), electric_field_y.reshape(old_shape), electric_field_z.reshape(old_shape)
