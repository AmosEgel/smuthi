# -*- coding: utf-8 -*-
import numpy as np
import smuthi.coordinates as coord
import smuthi.post_processing as pp
import smuthi.vector_wave_functions as vwf
import smuthi.index_conversion as idx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection


def show_scattered_field(xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0, resolution=25, interpolate=5, n_effective=None,
                         azimuthal_angles=None, vacuum_wavelength=None, particle_collection=None, linear_system=None,
                         layer_system=None, layerresponse_precision=None, max_field=None):
    if xmin == xmax:
        yvec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        zvec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        yarr, zarr = np.meshgrid(yvec, zvec)
        xarr = yarr - yarr
    elif ymin == ymax:
        xvec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        zvec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        xarr, zarr = np.meshgrid(xvec, zvec)
        yarr = xarr - xarr
    else:
        xvec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        yvec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        xarr, yarr = np.meshgrid(xvec, yvec)
        zarr = xarr - xarr

    e_x, e_y, e_z = scattered_electric_field(xarr, yarr, zarr, n_effective, azimuthal_angles, vacuum_wavelength,
                                             particle_collection, linear_system, layer_system, layerresponse_precision)

    if xmin == xmax:
        #plt.subplots(1, 3, sharey=True)
        plt.subplot(1, 3, 1)
        if max_field is None:
            plt.pcolormesh(yarr, zarr, e_x.real)
        else:
            plt.pcolormesh(yarr, zarr, e_x.real, vmin=-max_field, vmax=max_field)
        for il in range(1, layer_system.number_of_layers()):
            plt.plot([ymin, ymax], [layer_system.reference_z(il), layer_system.reference_z(il)], 'w')
        patches = []
        for particle in particle_collection.particles:
            if particle['shape'] == 'sphere':
                pos = particle['position']
                patches.append(Circle((pos[1], pos[2]), particle['radius']))
        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('x-component of electric field')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        if max_field is None:
            plt.pcolormesh(yarr, zarr, e_y.real)
        else:
            plt.pcolormesh(yarr, zarr, e_y.real, vmin=-max_field, vmax=max_field)
        for il in range(1, layer_system.number_of_layers()):
            plt.plot([ymin, ymax], [layer_system.reference_z(il), layer_system.reference_z(il)], 'w')
        patches = []
        for particle in particle_collection.particles:
            if particle['shape'] == 'sphere':
                pos = particle['position']
                patches.append(Circle((pos[1], pos[2]), particle['radius']))
        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('y-component of electric field')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        if max_field is None:
            plt.pcolormesh(yarr, zarr, e_z.real)
        else:
            plt.pcolormesh(yarr, zarr, e_z.real, vmin=-max_field, vmax=max_field)
        for il in range(1, layer_system.number_of_layers()):
            plt.plot([ymin, ymax], [layer_system.reference_z(il), layer_system.reference_z(il)], 'w')
        patches = []
        for particle in particle_collection.particles:
            if particle['shape'] == 'sphere':
                pos = particle['position']
                patches.append(Circle((pos[1], pos[2]), particle['radius']))
        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('z-component of electric field')
        plt.colorbar()

    elif ymin == ymax:
        plt.figure()
        ax1 = plt.subplot(131)
        if max_field is None:
            plt.pcolormesh(xarr, zarr, e_x.real)
        else:
            plt.pcolormesh(xarr, zarr, e_x.real, vmin=-max_field, vmax=max_field)
        for il in range(1, layer_system.number_of_layers()):
            plt.plot([xmin, xmax], [layer_system.reference_z(il), layer_system.reference_z(il)], 'w')
        patches = []
        for particle in particle_collection.particles:
            if particle['shape'] == 'sphere':
                pos = particle['position']
                patches.append(Circle((pos[0], pos[2]), particle['radius']))
        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('x-component of electric field')
        plt.colorbar()

        ax2 = plt.subplot(132, sharey=ax1)
        if max_field is None:
            plt.pcolormesh(xarr, zarr, e_y.real)
        else:
            plt.pcolormesh(xarr, zarr, e_y.real, vmin=-max_field, vmax=max_field)
        for il in range(1, layer_system.number_of_layers()):
            plt.plot([xmin, xmax], [layer_system.reference_z(il), layer_system.reference_z(il)], 'w')
        patches = []
        for particle in particle_collection.particles:
            if particle['shape'] == 'sphere':
                pos = particle['position']
                patches.append(Circle((pos[0], pos[2]), particle['radius']))
        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('y-component of electric field')
        plt.colorbar()

        ax3 = plt.subplot(133, sharey=ax1)
        if max_field is None:
            plt.pcolormesh(xarr, zarr, e_z.real)
        else:
            plt.pcolormesh(xarr, zarr, e_z.real, vmin=-max_field, vmax=max_field)
        for il in range(1, layer_system.number_of_layers()):
            plt.plot([xmin, xmax], [layer_system.reference_z(il), layer_system.reference_z(il)], 'w')
        patches = []
        for particle in particle_collection.particles:
            if particle['shape'] == 'sphere':
                pos = particle['position']
                patches.append(Circle((pos[0], pos[2]), particle['radius']))
        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('z-component of electric field')
        plt.colorbar()

    else:
        plt.figure()
        plt.subplot(1, 3, 1)
        if max_field is None:
            plt.pcolormesh(xarr, yarr, e_x.real)
        else:
            plt.pcolormesh(xarr, yarr, e_x.real, vmin=-max_field, vmax=max_field)
        patches = []
        for particle in particle_collection.particles:
            if particle['shape'] == 'sphere':
                pos = particle['position']
                patches.append(Circle((pos[0], pos[1]), particle['radius']))
        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('x-component of electric field')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        if max_field is None:
            plt.pcolormesh(xarr, yarr, e_y.real)
        else:
            plt.pcolormesh(xarr, yarr, e_y.real, vmin=-max_field, vmax=max_field)
        patches = []
        for particle in particle_collection.particles:
            if particle['shape'] == 'sphere':
                pos = particle['position']
                patches.append(Circle((pos[0], pos[1]), particle['radius']))
        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('y-component of electric field')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        if max_field is None:
            plt.pcolormesh(xarr, yarr, e_z.real)
        else:
            plt.pcolormesh(xarr, yarr, e_z.real, vmin=-max_field, vmax=max_field)
        patches = []
        for particle in particle_collection.particles:
            if particle['shape'] == 'sphere':
                pos = particle['position']
                patches.append(Circle((pos[0], pos[1]), particle['radius']))
        p = PatchCollection(patches)
        p.set_facecolor('w')
        p.set_edgecolor('k')
        plt.gca().add_collection(p)
        plt.gca().set_aspect("equal")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('z-component of electric field')
        plt.colorbar()

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
            pwp_rs = pp.plane_wave_pattern_rs(n_effective=n_effective, azimuthal_angles=azimuthal_angles,
                                              vacuum_wavelength=vacuum_wavelength,
                                              particle_collection=particle_collection, linear_system=linear_system,
                                              layer_system=layer_system, layer_numbers=[il],
                                              layerresponse_precision=layerresponse_precision)[0]

            for pol in range(2):
                integrand_alpha_x = np.zeros((len(azimuthal_angles), len(layer_indices)), dtype=complex)
                integrand_alpha_y = np.zeros((len(azimuthal_angles), len(layer_indices)), dtype=complex)
                integrand_alpha_z = np.zeros((len(azimuthal_angles), len(layer_indices)), dtype=complex)

                for ja, a in enumerate(azimuthal_angles):
                    print(ja)
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
