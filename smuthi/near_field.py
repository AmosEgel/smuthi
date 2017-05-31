# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as interp
import smuthi.coordinates as coord
import smuthi.plane_wave_pattern as pwp
import smuthi.vector_wave_functions as vwf
import smuthi.index_conversion as idx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection


def show_scattered_field(quantities_to_plot, xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0, resolution=25,
                         interpolate=None, n_effective=None, azimuthal_angles=None, vacuum_wavelength=None,
                         particle_collection=None, linear_system=None, layer_system=None, layerresponse_precision=None,
                         max_field=None, length_unit='length unit', max_particle_distance=float('inf')):
    """Plot the electric near field along a plane. To plot along the xy-plane, specify zmin=zmax and so on.

    input:
    quantities_to_plot:         List of strings that specify what to plot. Select from 'E_x', 'E_y', 'E_z', 'norm(E)'
                                Example: ('E_x', 'norm(E)') generates two plots, showing the real part of the electric
                                field's x component as well as the norm of the E-vector.
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
    vacuum_wavelength:          Vacuum wavelength
    particle_collection:        smuthi.particle.ParticleCollection object
    linear_system:              smuthi.linear_system.LinearSystem object
    layer_system:               smuthi.layers.LayerSystem object
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    max_field:                  If specified, truncate the color scale of the field plots at that value.
    max_particle_distance       Show particles that are closer than that distance to the image plane
                                (length unit, default = inf).
    """
    if xmin == xmax:
        dim1vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        yarr, zarr = np.meshgrid(dim1vec, dim2vec)
        xarr = yarr - yarr + xmin
        dim1name = 'y (' + length_unit + ')'
        dim2name = 'z (' + length_unit + ')'
    elif ymin == ymax:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        xarr, zarr = np.meshgrid(dim1vec, dim2vec)
        yarr = xarr - xarr + ymin
        dim1name = 'x (' + length_unit + ')'
        dim2name = 'z (' + length_unit + ')'
    else:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        xarr, yarr = np.meshgrid(dim1vec, dim2vec)
        zarr = xarr - xarr + zmin
        dim1name = 'x (' + length_unit + ')'
        dim2name = 'y (' + length_unit + ')'

    e_x_raw, e_y_raw, e_z_raw = scattered_electric_field(xarr, yarr, zarr, n_effective, azimuthal_angles,
                                                         vacuum_wavelength, particle_collection, linear_system,
                                                         layer_system, layerresponse_precision)

    if interpolate is None:
        e_x, e_y, e_z = e_x_raw, e_y_raw, e_z_raw
        dim1vecfine = dim1vec
        dim2vecfine = dim2vec
    else:
        dim1vecfine = np.linspace(dim1vec[0], dim1vec[-1], (dim1vec[-1] - dim1vec[0]) / interpolate + 1, endpoint=True)
        dim2vecfine = np.linspace(dim2vec[0], dim2vec[-1], (dim2vec[-1] - dim2vec[0]) / interpolate + 1, endpoint=True)

        real_ex_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_raw.real)
        imag_ex_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_x_raw.imag)
        e_x = real_ex_interpolant(dim2vecfine, dim1vecfine) + 1j * imag_ex_interpolant(dim2vecfine, dim1vecfine)

        real_ey_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_raw.real)
        imag_ey_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_y_raw.imag)
        e_y = real_ey_interpolant(dim2vecfine, dim1vecfine) + 1j * imag_ey_interpolant(dim2vecfine, dim1vecfine)

        real_ez_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_raw.real)
        imag_ez_interpolant = interp.RectBivariateSpline(dim2vec, dim1vec, e_z_raw.imag)
        e_z = real_ez_interpolant(dim2vecfine, dim1vecfine) + 1j * imag_ez_interpolant(dim2vecfine, dim1vecfine)

    if max_field is None:
        vmin = None
        vmax = None
    else:
        vmin = -max_field
        vmax = max_field

    for quantity in quantities_to_plot:
        plt.figure()
        if quantity == 'E_x':
            plt.pcolormesh(dim1vecfine, dim2vecfine, e_x.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
            plt.title('x-component of electric field')
        elif quantity == 'E_y':
            plt.pcolormesh(dim1vecfine, dim2vecfine, e_y.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
            plt.title('y-component of electric field')
        elif quantity == 'E_z':
            plt.pcolormesh(dim1vecfine, dim2vecfine, e_z.real, vmin=vmin, vmax=vmax, cmap='RdYlBu')
            plt.title('z-component of electric field')
        elif quantity == 'norm(E)':
            plt.pcolormesh(dim1vecfine, dim2vecfine,
                           np.sqrt(abs(e_x) ** 2 + abs(e_y) ** 2 + abs(e_z) ** 2), vmin=0, vmax=vmax, cmap='inferno')
            plt.title('norm of electric field')

        plt.colorbar()
        plt.xlabel(dim1name)
        plt.ylabel(dim2name)

        # plot layer interfaces
        if not zmin == zmax:
            for il in range(1, layer_system.number_of_layers()):
                plt.plot([dim1vec[0], dim1vec[-1]], [layer_system.reference_z(il), layer_system.reference_z(il)], 'w')

        # plot particles
        patches = []
        for particle in particle_collection.particles:
            pos = particle['position']
            if xmin == xmax:
                if abs(xmin - pos[0]) < max_particle_distance:
                    if particle['shape'] == 'sphere':
                        patches.append(Circle((pos[1], pos[2]), particle['radius']))
                    elif particle['shape'] == 'spheroid':
                        if not particle['euler angles'] == [0, 0, 0]:
                            raise ValueError('rotated particles currently not supported')
                        patches.append(Ellipse(xy=(pos[1], pos[2]), width=2*particle['semi axis a'],
                                               height=2*particle['semi axis c']))
                    elif particle['shape'] == 'finite cylinder':
                        cylinder_radius = particle['cylinder radius']
                        cylinder_height = particle['cylinder height']
                        patches.append(Rectangle((pos[1] - cylinder_radius, pos[2] - cylinder_height / 2),
                                                 2 * cylinder_radius, cylinder_height))

            elif ymin == ymax:
                if abs(ymin - pos[1]) < max_particle_distance:
                    if particle['shape'] == 'sphere':
                        patches.append(Circle((pos[0], pos[2]), particle['radius']))
                    elif particle['shape'] == 'spheroid':
                        if not particle['euler angles'] == [0, 0, 0]:
                            raise ValueError('rotated particles currently not supported')
                        patches.append(Ellipse(xy=(pos[0], pos[2]), width=2*particle['semi axis a'],
                                               height=2*particle['semi axis c']))
                    elif particle['shape'] == 'finite cylinder':
                        cylinder_radius = particle['cylinder radius']
                        cylinder_height = particle['cylinder height']
                        patches.append(Rectangle((pos[0] - cylinder_radius, pos[2] - cylinder_height / 2),
                                                 2 * cylinder_radius, cylinder_height))

            elif zmin == zmax:
                if abs(zmin - pos[2]) < max_particle_distance:
                    if particle['shape'] == 'sphere':
                        patches.append(Circle((pos[0], pos[1]), particle['radius']))
                    elif particle['shape'] == 'spheroid':
                        if not particle['euler angles'] == [0, 0, 0]:
                            raise ValueError('rotated particles currently not supported')
                        patches.append(Circle((pos[0], pos[1]), particle['semi axis a']))
                    elif particle['shape'] == 'finite cylinder':
                        cylinder_radius = particle['cylinder radius']
                        patches.append(Circle((pos[0], pos[1]), cylinder_radius))

        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")

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
