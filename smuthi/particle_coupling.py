# -*- coding: utf-8 -*-
"""Provide routines for multiple scattering."""

import numpy as np
import scipy.special
import smuthi.coordinates as coord
import smuthi.layers as lay
import smuthi.spherical_functions as sf
import smuthi.field_expansion as fldex
import smuthi.vector_wave_functions as vwf
import matplotlib.pyplot as plt


radial_distance_array = None
radial_distance_table = None
radial_particle_coupling_lookup = None


def layer_mediated_coupling_block(vacuum_wavelength, receiving_particle, emitting_particle, layer_system, 
                                  k_parallel='default', show_integrand=False):
    """Layer-system mediated particle coupling matrix :math:`W^R` for two particles. This routine is explicit, but slow.

    Args:
        vacuum_wavelength (float):                          Vacuum wavelength :math:`\lambda` (length unit)
        receiving_particle (smuthi.particles.Particle):     Particle that receives the scattered field
        emitting_particle (smuthi.particles.Particle):      Particle that emits the scattered field
        layer_system (smuthi.layers.LayerSystem):           Stratified medium in which the coupling takes place
        k_parallel (numpy ndarray):                         In-plane wavenumbers for Sommerfeld integral
                                                            If 'default', use smuthi.coordinates.default_k_parallel
        show_integrand (bool):                              If True, the norm of the integrand is plotted.

    Returns:
        Layer mediated coupling matrix block as numpy array.
    """
    if type(k_parallel) == str and k_parallel == 'default':
        k_parallel = coord.default_k_parallel
        
    omega = coord.angular_frequency(vacuum_wavelength)

    # index specs
    lmax1 = receiving_particle.l_max
    mmax1 = receiving_particle.m_max
    lmax2 = emitting_particle.l_max
    mmax2 = emitting_particle.m_max
    blocksize1 = fldex.blocksize(lmax1, mmax1)
    blocksize2 = fldex.blocksize(lmax2, mmax2)

    # cylindrical coordinates of relative position vectors
    rs1 = np.array(receiving_particle.position)
    rs2 = np.array(emitting_particle.position)
    rs2s1 = rs1 - rs2
    rhos2s1 = np.linalg.norm(rs2s1[0:2])
    phis2s1 = np.arctan2(rs2s1[1], rs2s1[0])
    is1 = layer_system.layer_number(rs1[2])
    ziss1 = rs1[2] - layer_system.reference_z(is1)
    is2 = layer_system.layer_number(rs2[2])
    ziss2 = rs2[2] - layer_system.reference_z(is2)

    # wave numbers
    kis1 = omega * layer_system.refractive_indices[is1]
    kis2 = omega * layer_system.refractive_indices[is2]
    kzis1 = coord.k_z(k_parallel=k_parallel, k=kis1)
    kzis2 = coord.k_z(k_parallel=k_parallel, k=kis2)

    # phase factors
    ejkz = np.zeros((2, 2, len(k_parallel)), dtype=complex)  # indices are: particle, plus/minus, kpar_idx
    ejkz[0, 0, :] = np.exp(1j * kzis1 * ziss1)
    ejkz[0, 1, :] = np.exp(- 1j * kzis1 * ziss1)
    ejkz[1, 0, :] = np.exp(1j * kzis2 * ziss2)
    ejkz[1, 1, :] = np.exp(- 1j * kzis2 * ziss2)

    # layer response
    L = np.zeros((2, 2, 2, len(k_parallel)), dtype=complex)  # polarization, pl/mn1, pl/mn2, kpar_idx
    for pol in range(2):
        L[pol, :, :, :] = lay.layersystem_response_matrix(pol, layer_system.thicknesses,
                                                          layer_system.refractive_indices, k_parallel, omega, is2, is1)

    # transformation coefficients
    B = [np.zeros((2, 2, blocksize1, len(k_parallel)), dtype=complex), 
         np.zeros((2, 2, blocksize2, len(k_parallel)), dtype=complex)]
    # list index: particle, np indices: pol, plus/minus, n, kpar_idx

    m_vec = [np.zeros(blocksize1, dtype=int), np.zeros(blocksize2, dtype=int)]
    
    # precompute spherical functions
    ct = kzis1 / kis1
    st = k_parallel / kis1
    _, pilm_list_pl, taulm_list_pl = sf.legendre_normalized(ct, st, lmax1)
    _, pilm_list_mn, taulm_list_mn = sf.legendre_normalized(-ct, st, lmax1)
    pilm = (pilm_list_pl, pilm_list_mn)
    taulm = (taulm_list_pl, taulm_list_mn)
    
    for tau in range(2):
        for m in range(-mmax1, mmax1 + 1):
            for l in range(max(1, abs(m)), lmax1 + 1):
                n = fldex.multi_to_single_index(tau, l, m, lmax1, mmax1)
                m_vec[0][n] = m
                for iplmn in range(2):
                    for pol in range(2):
                        B[0][pol, iplmn, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm[iplmn],
                                                                                     taulm_list=taulm[iplmn], dagger=True)
    
    ct = kzis2 / kis2
    st = k_parallel / kis2
    _, pilm_list_pl, taulm_list_pl = sf.legendre_normalized(ct, st, lmax2)
    _, pilm_list_mn, taulm_list_mn = sf.legendre_normalized(-ct, st, lmax2)
    pilm = (pilm_list_pl, pilm_list_mn)
    taulm = (taulm_list_pl, taulm_list_mn)
                        
    for tau in range(2):
        for m in range(-mmax2, mmax2 + 1):
            for l in range(max(1, abs(m)), lmax2 + 1):
                n = fldex.multi_to_single_index(tau, l, m, lmax2, mmax2)
                m_vec[1][n] = m
                for iplmn in range(2):
                    for pol in range(2):
                        B[1][pol, iplmn, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm[iplmn],
                                                                                     taulm_list=taulm[iplmn], dagger=False)

    BeL = np.zeros((2, 2, blocksize1, len(k_parallel)), dtype=complex)  # indices are: pol, plmn2, n1, kpar_idx
    for iplmn1 in range(2):
        for pol in range(2):
            BeL[pol, :, :, :] += (L[pol, iplmn1, :, np.newaxis, :] * B[0][pol, iplmn1, np.newaxis, :, :]
                                  * ejkz[0, iplmn1, :])
    BeLBe = np.zeros((blocksize1, blocksize2, len(k_parallel)), dtype=complex)  # indices are: n1, n2, kpar_idx
    for iplmn2 in range(2):
        for pol in range(2):
            BeLBe += BeL[pol, iplmn2, :, np.newaxis, :] * B[1][pol, iplmn2, :, :] * ejkz[1, 1 - iplmn2, :]

    # bessel function and jacobi factor
    bessel_list = []
    for dm in range(lmax1 + lmax2 + 1):
        bessel_list.append(scipy.special.jv(dm, k_parallel * rhos2s1))
    bessel_full = np.array([[bessel_list[abs(m_vec[0][n1] - m_vec[1][n2])]
                             for n2 in range(blocksize2)] for n1 in range(blocksize1)])
    jacobi_vector = k_parallel / (kzis2 * kis2)
    integrand = bessel_full * jacobi_vector * BeLBe
    integral = np.trapz(integrand, x=k_parallel, axis=-1)
    m2_minus_m1 = m_vec[1] - m_vec[0][np.newaxis].T
    wr = 4 * (1j) ** abs(m2_minus_m1) * np.exp(1j * m2_minus_m1 * phis2s1) * integral

    if show_integrand:
        norm_integrand = np.zeros(len(k_parallel))
        for i in range(len(k_parallel)):
            norm_integrand[i] = 4 * np.linalg.norm(integrand[:, :, i])
        plt.plot(k_parallel.real / omega, norm_integrand)
        plt.show()

    return wr


def layer_mediated_coupling_matrix(vacuum_wavelength, particle_list, layer_system, k_parallel='default'):
    """Layer system mediated particle coupling matrix W^R for a particle collection in a layered medium.

    Args:
        vacuum_wavelength (float):                                  Wavelength in length unit
        particle_list (list of smuthi.particles.Particle obejcts:   Scattering particles
        layer_system (smuthi.layers.LayerSystem):                   The stratified medium
        k_parallel (numpy.ndarray or str):                          In-plane wavenumber for Sommerfeld integrals. 
                                                                    If 'default', smuthi.coordinates.default_k_parallel
    
    Returns:
        Ensemble coupling matrix as numpy array.
    """
    
    # indices
    blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in particle_list]

    # initialize result
    wr = np.zeros((sum(blocksizes), sum(blocksizes)), dtype=complex)

    for s1, particle1 in enumerate(particle_list):
        idx1 = np.array(range(sum(blocksizes[:s1]), sum(blocksizes[:s1]) + blocksizes[s1]))
        for s2, particle2 in enumerate(particle_list):
            idx2 = range(sum(blocksizes[:s2]), sum(blocksizes[:s2]) + blocksizes[s2])
            wr[idx1[:, None], idx2] = layer_mediated_coupling_block(vacuum_wavelength, particle1, particle2,
                                                                    layer_system, k_parallel)

    return wr


def direct_coupling_block(vacuum_wavelength, receiving_particle, emitting_particle, layer_system):
    """Direct particle coupling matrix :math:`W` for two particles. This routine is explicit, but slow.

    Args:
        vacuum_wavelength (float):                          Vacuum wavelength :math:`\lambda` (length unit)
        receiving_particle (smuthi.particles.Particle):     Particle that receives the scattered field
        emitting_particle (smuthi.particles.Particle):      Particle that emits the scattered field
        layer_system (smuthi.layers.LayerSystem):           Stratified medium in which the coupling takes place

    Returns:
        Direct coupling matrix block as numpy array.
    """
    omega = coord.angular_frequency(vacuum_wavelength)

    # index specs
    lmax1 = receiving_particle.l_max
    mmax1 = receiving_particle.m_max
    lmax2 = emitting_particle.l_max
    mmax2 = emitting_particle.m_max
    blocksize1 = fldex.blocksize(lmax1, mmax1)
    blocksize2 = fldex.blocksize(lmax2, mmax2)

    # initialize result
    w = np.zeros((blocksize1, blocksize2), dtype=complex)

    # check if particles are in same layer
    rS1 = receiving_particle.position
    rS2 = emitting_particle.position
    iS1 = layer_system.layer_number(rS1[2])
    iS2 = layer_system.layer_number(rS2[2])
    if iS1 == iS2 and not emitting_particle == receiving_particle:
        k = omega * layer_system.refractive_indices[iS1]
        dx = rS1[0] - rS2[0]
        dy = rS1[1] - rS2[1]
        dz = rS1[2] - rS2[2]
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        cos_theta = dz / d
        sin_theta = np.sqrt(dx**2 + dy**2) / d
        phi = np.arctan2(dy, dx)

        # spherical functions
        bessel_h = [sf.spherical_hankel(n, k * d) for n in range(lmax1 + lmax2 + 1)]
        legendre, _, _ = sf.legendre_normalized(cos_theta, sin_theta, lmax1 + lmax2)

        for m1 in range(-mmax1, mmax1 + 1):
            for m2 in range(-mmax2, mmax2 + 1):
                eimph = np.exp(1j * (m2 - m1) * phi)
                for l1 in range(max(1, abs(m1)), lmax1 + 1):
                    for l2 in range(max(1, abs(m2)), lmax2 + 1):
                        A, B = complex(0), complex(0)
                        for ld in range(max(abs(l1 - l2), abs(m1 - m2)), l1 + l2 + 1):  # if ld<abs(m1-m2) then P=0
                            a5, b5 = vwf.ab5_coefficients(l2, m2, l1, m1, ld)
                            A += a5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                            B += b5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                        A, B = eimph * A, eimph * B
                        for tau1 in range(2):
                            n1 = fldex.multi_to_single_index(tau1, l1, m1, lmax1, mmax1)
                            for tau2 in range(2):
                                n2 = fldex.multi_to_single_index(tau2, l2, m2, lmax2, mmax2)
                                if tau1 == tau2:
                                    w[n1, n2] = A  # remember that w = A.T
                                else:
                                    w[n1, n2] = B

    return w


def direct_coupling_matrix(vacuum_wavelength, particle_list, layer_system):
    """Return the direct particle coupling matrix W for a particle collection in a layered medium.

    Args:
        vacuum_wavelength (float):                                  Wavelength in length unit
        particle_list (list of smuthi.particles.Particle obejcts:   Scattering particles
        layer_system (smuthi.layers.LayerSystem):                   The stratified medium
    
    Returns:
        Ensemble coupling matrix as numpy array.
    """
    # indices
    blocksizes = [fldex.blocksize(particle.l_max, particle.m_max)
                  for particle in particle_list]

    # initialize result
    w = np.zeros((sum(blocksizes), sum(blocksizes)), dtype=complex)

    for s1, particle1 in enumerate(particle_list):
        idx1 = np.array(range(sum(blocksizes[:s1]), sum(blocksizes[:s1+1])))
        for s2, particle2 in enumerate(particle_list):
            idx2 = range(sum(blocksizes[:s2]), sum(blocksizes[:s2+1]))
            w[idx1[:, None], idx2] = direct_coupling_block(vacuum_wavelength, particle1, particle2, layer_system)

    return w


def radial_coupling_lookup(vacuum_wavelength, particle_list, layer_system, k_parallel='default', resolution=None):
    
    global radial_distance_array
    global radial_distance_table
    global particle_coupling_lookup_2D
    
    if resolution is None:
        resolution = vacuum_wavelength / 100
    
    l_max = max([particle.l_max for particle in particle_list])
    m_max = max([particle.m_max for particle in particle_list])
    n_max = fldex.blocksize(l_max, m_max)
    max_rho = max([np.sqrt(particle.position[0] ** 2 + particle.position[1] ** 2) for particle in particle_list])
    
    radial_distance_array = np.arange(0, 2 * (max_rho + resolution), resolution)
    
    radial_distance_table = []
    for i1, particle1 in enumerate(particle_list):
        radial_distance_table.append([])
        for i2, particle2 in enumerate(particle_list):
            radial_distance_table[i1].append(np.sqrt((particle1.position[0] - particle2.position[0])**2 
                                                     + (particle1.position[1] - particle2.position[1])**2))
    
    z = particle_list[0].position[2]
    i_s = layer_system.layer_number(z)
    k_is = layer_system.wavenumber(i_s, vacuum_wavelength)
    dz = z - layer_system.reference_z(i_s) 
        
    lookup_table = np.zeros((n_max, n_max, len(radial_distance_array)), dtype=complex)
    
    # direct -----------------------------------------------------------------------------------------------------------
    w = np.zeros((n_max, n_max, len(radial_distance_array)), dtype=complex)

    ct = 0
    st = 1
    bessel_h = [sf.spherical_hankel(n, k * radial_distance_array) for n in range(2* l_max + 1)]
    legendre, _, _ = sf.legendre_normalized(ct, st, 2 * l_max)

    for m1 in range(-m_max, m_max + 1):
        for m2 in range(-m_max, m_max + 1):
            for l1 in range(max(1, abs(m1)), l_max + 1):
                for l2 in range(max(1, abs(m2)), l_max + 1):
                    A = np.zeros(len(radial_distance_array), dtype=complex)
                    B = np.zeros(len(radial_distance_array), dtype=complex)
                    for ld in range(max(abs(l1 - l2), abs(m1 - m2)), l1 + l2 + 1):  # if ld<abs(m1-m2) then P=0
                        a5, b5 = vwf.ab5_coefficients(l2, m2, l1, m1, ld)
                        A += a5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                        B += b5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                    for tau1 in range(2):
                        n1 = fldex.multi_to_single_index(tau1, l1, m1, l_max, m_max)
                        for tau2 in range(2):
                            n2 = fldex.multi_to_single_index(tau2, l2, m2, l_max, m_max)
                            if tau1 == tau2:
                                w[n1, n2, :] = A  # remember that w = A.T
                            else:
                                w[n1, n2, :] = B
    
    # layer mediated ---------------------------------------------------------------------------------------------------
    if type(k_parallel) == str and k_parallel == 'default':
        k_parallel = coord.default_k_parallel
    kz_is = coord.k_z(k_parallel=k_parallel, k=k_is)
    len_kp = len(k_parallel)
    wr_lookup_integrand = np.ones((n_max, n_max, len(radial_distance_array), len_kp), dtype=complex)
    
    # phase factors
    ejkz = np.zeros((2, len_k), dtype=complex)  # pl/mn, kp
    ejkz[0, :] = np.exp(1j * kz_is * dz)
    ejkz[1, :] = np.exp(- 1j * kz_is * dz)
        
    # layer response
    L = np.zeros((2, 2, 2, 1, 1, len_kp), dtype=complex)  # pol, pl/mn1, pl/mn2, kp
    for pol in range(2):
        L[pol, :, :, :] = lay.layersystem_response_matrix(pol, layer_system.thicknesses, 
                                                          layer_system.refractive_indices, k_parallel,
                                                          coord.angular_frequency(vacuum_wavelength), is2, is1)
    
    # transformation coefficients
    B_dag = np.zeros((2, 2, n_max, len_kp), dtype=complex)  # pol, pl/mn, n, kp
    B = np.zeros((2, 2, n_max, len_kp), dtype=complex)  # pol, pl/mn, n, kp
    ct = kz_is / k_is
    st = k_parallel / k_is
    _, pilm_pl, taulm_pl = sf.legendre_normalized(ct, st, lmax1)
    _, pilm_mn, taulm_mn = sf.legendre_normalized(-ct, st, lmax1)
    pilm = (pilm_list_pl, pilm_list_mn)
    taulm = (taulm_list_pl, taulm_list_mn)
    
    for tau in range(2):
        for m in range(-m_max, m_max + 1):
            for l in range(max(1, abs(m)), l_max + 1):
                n = fldex.multi_to_single_index(tau, l, m, l_max, m_max)
                m_vec[n] = m
                for pol in range(2):
                    B_dag[pol, 0, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl, 
                                                                                    taulm_list=taulm_pl, dagger=True)
                    B_dag[pol, 1, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn, 
                                                                                    taulm_list=taulm_mn, dagger=True)
                    B[pol, 0, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl, 
                                                                             taulm_list=taulm_pl, dagger=False)
                    B[pol, 1, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn, 
                                                                             taulm_list=taulm_mn, dagger=False)
                    
    bel = np.zeros((2, 2, n_max, len_k), dtype=complex)  # pol, pl/mn2, n1, kp
    for pm1 in range(2):
        for pol in range(2):
            bl[pol, :, :, :] += L[pol, pm1,  :, None, :] * B_dag[pol, pm1, None, :, :] * ejkz[pm1, None, None, :]
                                  
    belbe = np.zeros((n_max, n_max, len_k), dtype=complex)  # n1, n2, kp
    for pm2 in range(2):
        for pol in range(2):
            belbe += bel[pol, pm2, :, None, :] * B[pol, pm2, None, :, :] * ejkz[1 - pm2, None, None, :] 
                                
    # bessel function and jacobi factor
    bessel_list = []
    for dm in range(2 * l_max + 1):
        bessel_list.append(scipy.special.jv(dm, k_parallel[None, :] * radial_distance_array[:, None]))
    bessel_full = np.array([[bessel_list[abs(m_vec[n1] - m_vec[n2])] for n2 in range(n_max)] for n1 in range(n_max)])
    jacobi_vector = k_parallel / (kz_is * k_is)
    integrand = bessel_full * jacobi_vector[None, None, None, :] * belbe[:, :, None, :]  # n1, n2, rho, kp
    integral = np.trapz(integrand, x=k_parallel, axis=-1)  # n1, n2, rho
    m2_minus_m1 = m_vec[None, :, None] - m_vec[:, None, None]
    wr = 4 * (1j) ** abs(m2_minus_m1) * integral
    
    return w + wr