# -*- coding: utf-8 -*-
"""Provide routines for multiple scattering."""

import numpy as np
import scipy.special
import smuthi.index_conversion as idx
import smuthi.coordinates as coord
import smuthi.layers as lay
import smuthi.vector_wave_functions as vwf
import smuthi.spherical_functions as sf
import matplotlib.pyplot as plt
import warnings


def layer_mediated_coupling_block(vacuum_wavelength, receiving_particle_position, emitting_particle_position,
                                  layer_system, neff_contour, layerresponse_precision=None, show_integrand=False):
    """Return the layer-system mediated particle coupling matrix W^R for two particles. This routine is explicit, but
    slow.

    Input:
    vacuum_wavelength:              (length unit)
    receiving_particle_position:    In the format [x,y,z] (length unit)
    emitting_particle_position:     In the format [x,y,z] (length unit)
    layer_system:                   An instance of smuthi.layers.LayerSystem describing the stratified medium
    neff_contour:                   An instance of smuthi.coordinates.ComplexContour to define the contour of the
                                    Sommerfeld integral
    layerresponse_precision:        Number of decimal digits (int). If specified, the layer-response is evaluated using
                                    mpmath multiple precision. Otherwise, standard numpy.
    show_integrand:                 If True, the norm of the integrand is plotted.
    """
    omega = coord.angular_frequency(vacuum_wavelength)

    # read out index specs
    lmax = idx.l_max
    mmax = idx.m_max
    blocksize = idx.number_of_indices()

    # cylindrical coordinates of relative position vectors
    rs1 = np.array(receiving_particle_position)
    rs2 = np.array(emitting_particle_position)
    rs2s1 = rs1 - rs2
    rhos2s1 = np.linalg.norm(rs2s1[0:2])
    phis2s1 = np.arctan2(rs2s1[1], rs2s1[0])
    is1 = layer_system.layer_number(rs1[2])
    ziss1 = rs1[2] - layer_system.reference_z(is1)
    is2 = layer_system.layer_number(rs2[2])
    ziss2 = rs2[2] - layer_system.reference_z(is2)

    # wave numbers
    neff = neff_contour.neff()
    kpar = omega * neff
    kis2 = omega * layer_system.refractive_indices[is2]
    kzis1 = coord.k_z(n_effective=neff, vacuum_wavelength=vacuum_wavelength,
                      refractive_index=layer_system.refractive_indices[is1])
    kzis2 = coord.k_z(n_effective=neff, vacuum_wavelength=vacuum_wavelength,
                      refractive_index=layer_system.refractive_indices[is2])

    # phase factors
    ejkz = np.zeros((2, 2, len(neff)), dtype=complex)  # indices are: particle, plus/minus, kpar_idx
    ejkz[0, 0, :] = np.exp(1j * kzis1 * ziss1)
    ejkz[0, 1, :] = np.exp(- 1j * kzis1 * ziss1)
    ejkz[1, 0, :] = np.exp(1j * kzis2 * ziss2)
    ejkz[1, 1, :] = np.exp(- 1j * kzis2 * ziss2)

    # layer response
    L = lay.evaluate_layerresponse_lookup(layer_system.thicknesses, layer_system.refractive_indices, kpar, omega, is2,
                                          is1, layerresponse_precision)  # polarization, pl/mn1, pl/mn2, kpar_idx

    # transformation coefficients
    B = np.zeros((2, 2, 2, blocksize, len(neff)), dtype=complex)  # indices are: particle, pol, plus/minus, n, kpar_idx

    m_vec = np.zeros(blocksize, dtype=int)
    kz_tup = (kzis1, kzis2)
    plmn_tup = (1, -1)
    dagger_tup = (True, False)

    for tau in range(2):
        for m in range(-mmax, mmax + 1):
            for l in range(max(1, abs(m)), lmax + 1):
                n = idx.multi_to_single_index(tau, l, m)
                m_vec[n] = m
                for iprt in range(2):
                    for iplmn, plmn in enumerate(plmn_tup):
                        for pol in range(2):
                            B[iprt, pol, iplmn, n, :] = vwf.transformation_coefficients_VWF(tau, l, m, pol, kpar,
                                                                                            plmn * kz_tup[iprt],
                                                                                            dagger=dagger_tup[iprt])

    BeL = np.zeros((2, 2, blocksize, len(neff)), dtype=complex)  # indices are: pol, plmn2, n1, kpar_idx
    for iplmn1 in range(2):
        for pol in range(2):
            BeL[pol, :, :, :] += (L[pol, iplmn1, :, np.newaxis, :] *
                                     B[0, pol, iplmn1, np.newaxis, :, :] * ejkz[0, iplmn1, :])
    BeLBe = np.zeros((blocksize, blocksize, len(neff)), dtype=complex)  # indices are: n1, n2, kpar_idx
    for iplmn2 in range(2):
        for pol in range(2):
            BeLBe += BeL[pol, iplmn2, :, np.newaxis, :] * B[1, pol, iplmn2, :, :] * ejkz[1, 1 - iplmn2, :]

    # bessel function and jacobi factor
    bessel_list = []
    for dm in range(2 * lmax + 1):
        bessel_list.append(scipy.special.jv(dm, kpar * rhos2s1))
    bessel_full = np.array([[bessel_list[abs(m_vec[n1] - m_vec[n2])]
                             for n1 in range(blocksize)] for n2 in range(blocksize)])
    jacobi_vector = kpar / (kzis2 * kis2)
    integrand = bessel_full * jacobi_vector * BeLBe
    integral = np.trapz(integrand, x=kpar, axis=-1)
    m2_minus_m1 = m_vec - m_vec[np.newaxis].T
    wr = 4 * (1j) ** abs(m2_minus_m1) * np.exp(1j * m2_minus_m1 * phis2s1) * integral

    if show_integrand:
        norm_integrand = np.zeros(len(neff))
        for i in range(len(neff)):
            norm_integrand[i] = 4 * np.linalg.norm(integrand[:, :, i])
        plt.plot(neff.real, norm_integrand)
        plt.show()

    return wr


def layer_mediated_coupling_matrix(vacuum_wavelength, particle_collection, layer_system, neff_contour,
                                   layerresponse_precision=None):
    """Return the layer-system mediated particle coupling matrix W^R for a particle collection.
    This routine is explicit, but slow. It is thus suited for problems with few particles only.

    NOT TESTED

    Input:
    vacuum_wavelength:              (length unit)
    particle_collection:            An instance of  smuthi.particles.ParticleCollection describing the scattering
                                    particles
    layer_system:                   An instance of smuthi.layers.LayerSystem describing the stratified medium
    neff_contour:                   An instance of smuthi.coordinates.ComplexContour to define the contour of the
                                    Sommerfeld integral
    layerresponse_precision:        Number of decimal digits (int). If specified, the layer-response is evaluated using
                                    mpmath multiple precision. Otherwise, standard numpy.
    """
    blocksize = idx.number_of_indices()
    particle_number = particle_collection.particle_number()

    # initialize result
    wr = np.zeros((particle_number, blocksize, particle_number, blocksize), dtype=complex)

    for s1, particle1 in enumerate(particle_collection.particles):
        rs1 = particle1['position']
        for s2, particle2 in enumerate(particle_collection.particles):
            rs2 = particle2['position']
            wrblock = layer_mediated_coupling_block(vacuum_wavelength, rs1, rs2, layer_system, neff_contour,
                                                    layerresponse_precision)

            wr[s1, :, s2, :] = wrblock

    return wr


def direct_coupling_matrix(vacuum_wavelength, particle_collection, layer_system):
    """Return the direct particle coupling matrix W for a particle collection in a layered medium.

    NOT TESTED

    Input:
    vacuum_wavelength:              (length unit)
    particle_collection:            An instance of  smuthi.particles.ParticleCollection describing the scattering
                                    particles
    layer_system:                   An instance of smuthi.layers.LayerSystem describing the stratified medium
    """
    omega = coord.angular_frequency(vacuum_wavelength)

    # indices
    blocksize = idx.number_of_indices()
    particle_number = particle_collection.particle_number()
    lmax = idx.l_max
    mmax = idx.m_max

    # initialize result
    w = np.zeros((particle_number, blocksize, particle_number, blocksize), dtype=complex)

    # check which particles are in same layer
    particle_layer_indices = [[] for x in range(layer_system.number_of_layers())]
    for i_particle, particle in enumerate(particle_collection.particles):
        particle_layer_indices[layer_system.layer_number(particle['position'][2])].append(i_particle)

    # direct coupling inside each layer
    pos_array = np.array(particle_collection.particle_positions())
    for i_layer in range(layer_system.number_of_layers()):
        if len(particle_layer_indices[i_layer]) > 1:

            k = omega * layer_system.refractive_indices[i_layer]

            # coordinates
            x = pos_array[particle_layer_indices[i_layer], 0]
            y = pos_array[particle_layer_indices[i_layer], 1]
            z = pos_array[particle_layer_indices[i_layer], 2]
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dz = z[:, np.newaxis] - z[np.newaxis, :]
            d = np.sqrt(dx**2 + dy**2 + dz**2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
                cos_theta = dz / d
                sin_theta = np.sqrt(dx**2 + dy**2) / d
            phi = np.arctan2(dy, dx)

            npart_layer = len(x)
            w_layer = np.zeros((npart_layer, blocksize, npart_layer, blocksize), dtype=complex)
            # indices: receiv. part., receiv. idx, emit. part., emit. idx

            # spherical functions
            bessel_h = [sf.spherical_hankel(n, k * d) for n in range(2 * lmax + 1)]
            legendre, _, _ = sf.legendre_normalized(cos_theta, sin_theta, 2 * lmax)

            for m1 in range(-mmax, mmax + 1):
                for m2 in range(-mmax, mmax + 1):
                    eimph = np.exp(1j * (m1 - m2) * phi)
                    for l1 in range(max(1, abs(m1)), lmax + 1):
                        for l2 in range(max(1, abs(m2)), lmax + 1):
                            A, B = complex(0), complex(0)
                            for ld in range(max(abs(l1 - l2), abs(m1 - m2)), l1 + l2 + 1):  # if ld<abs(m1-m2) then P=0
                                a5, b5 = vwf.ab5_coefficients(l1, m1, l2, m2, ld)
                                A += a5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                                B += b5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                            A, B = eimph * A, eimph * B
                            for tau1 in range(2):
                                n1 = idx.multi_to_single_index(tau1, l1, m1)
                                for tau2 in range(2):
                                    n2 = idx.multi_to_single_index(tau2, l2, m2)
                                    if tau1 == tau2:
                                        w_layer[:, n2, :, n1] = A  # remember that w = A.T
                                    else:
                                        w_layer[:, n2, :, n1] = B

            for i1 in range(npart_layer):
                for i2 in range(npart_layer):
                    if not i1 == i2:
                        s1 = particle_layer_indices[i_layer][i1]
                        s2 = particle_layer_indices[i_layer][i2]
                        w[s1, :, s2, :] = w_layer[i1, :, i2, :]

    return w
