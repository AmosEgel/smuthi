# -*- coding: utf-8 -*-
"""Routines for multiple scattering. The first half of the module contains functions to explicitly compute the 
coupling matrix entries. The second half of the module contains functions for the preparation of lookup tables that 
are used to approximate the coupling matrices by interoplation."""

from numba import complex128,int64,jit
from scipy.signal.filter_design import bessel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.special
import smuthi.coordinates as coord
import smuthi.cuda_sources as cu
import smuthi.field_expansion as fldex
import smuthi.layers as lay
import smuthi.spherical_functions as sf
import smuthi.vector_wave_functions as vwf
import sys
try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule
    import pycuda.cumath
except:
    pass


@jit(complex128(complex128[:], complex128[:]),
     nopython=True, cache=True, nogil=True)
def numba_trapz(y, x):
    out = 0.0 + 0.0j
    #TODO implement some (optional) advanced summation?
    #e.g. https://github.com/nschloe/accupy/blob/master/accupy/sums.py
    #or better Sum2  from https://doi.org/10.1137/030601818 (Algorithm 4.4)
    #Note, that this may need to have exact summation for x and y, and exact product.
    for i in range( len(y) - 2 ):
        out += (x[i+1]-x[i]) * (y[i+1] + y[i])/2.0
    return out


@jit((complex128[:], complex128[:,:,:],
      complex128[:,:,:,:],complex128[:,:,:], int64),
     nopython=True,cache=True
     ,nogil=True
     # , parallel=True
)
def eval_BeLBe(BeLBe, BeL, B1, ejkz, n2):
    for k in range(len(BeLBe)):
        for iplmn2 in range(2):
            for pol in range(2):
                BeLBe[k] += BeL[pol, iplmn2, k] * B1[pol, iplmn2, n2, k
                                     ] * ejkz[1, 1 - iplmn2, k]
        

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

    # bessel function and jacobi factor
    bessel_list = []
    for dm in range(lmax1 + lmax2 + 1):
        bessel_list.append(scipy.special.jv(dm, k_parallel * rhos2s1))
    jacobi_vector = k_parallel / (kzis2 * kis2)
    m2_minus_m1 = m_vec[1] - m_vec[0][np.newaxis].T
    wr_const = 4 * (1j) ** abs(m2_minus_m1) * np.exp(1j * m2_minus_m1 * phis2s1) 

    integral = np.zeros((blocksize1, blocksize2), dtype=complex) 
    for n1 in range(blocksize1):
        BeL = np.zeros((2, 2, len(k_parallel)), dtype=complex)  # indices are: pol, plmn2, n1, kpar_idx
        for iplmn1 in range(2):
            for pol in range(2):
                BeL[pol, :, :] += (L[pol, iplmn1, :, :]
                                        * B[0][pol, iplmn1, n1, :]
                                        * ejkz[0, iplmn1, :])
        for n2 in range(blocksize2):
            bessel_full = bessel_list[abs(m_vec[0][n1] - m_vec[1][n2])]
            BeLBe = np.zeros((len(k_parallel)), dtype=complex)
            eval_BeLBe(BeLBe, BeL, B[1], ejkz, n2)
            integrand = bessel_full * jacobi_vector * BeLBe
            integral[n1,n2] = numba_trapz(integrand, k_parallel)
    wr = wr_const * integral

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
        
        # the particle coupling operator is the transpose of the SVWF translation operator
        # therefore, (l1,m1) and (l2,m2) are interchanged:
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
                                    w[n1, n2] = A
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




def volumetric_coupling_lookup_table(vacuum_wavelength, particle_list, layer_system, k_parallel='default', 
                                     resolution=None):
    """Prepare Sommerfeld integral lookup table to allow for a fast calculation of the coupling matrix by interpolation.
    This function is called when not all particles are on the same z-position.
    
    Args:
        vacuum_wavelength (float):  Vacuum wavelength in length units
        particle_list (list):       List of particle objects
        layer_system (smuthi.layers.LayerSystem):    Stratified medium
        k_parallel (numpy.ndarray or str):           In-plane wavenumber for Sommerfeld integrals.
                                                     If 'default', smuthi.coordinates.default_k_parallel
        resolution (float): Spatial resolution of lookup table in length units. (default: vacuum_wavelength / 100)        
                            Smaller means more accurate but higher memory footprint 
                            
    Returns:
        (tuple): tuple containing:

            w_pl (ndarray):  Coupling lookup for z1 + z2, indices are [rho, z, n1, n2]. Includes layer mediated coupling.
            w_mn (ndarray):  Coupling lookup for z1 + z2, indices are [rho, z, n1, n2]. Includes layer mediated and 
                             direct coupling.
            rho_array (ndarray):    Values for the radial distance considered for the lookup (starting from negative 
                                    numbers to allow for simpler cubic interpolation without distinction of cases 
                                    for lookup edges
            sz_array (ndarray):     Values for the sum of z-coordinates (z1 + z2) considered for the lookup
            dz_array (ndarray):     Values for the difference of z-coordinates (z1 - z2) considered for the lookup 
    """
    sys.stdout.write('Prepare 3D particle coupling lookup:\n')
    sys.stdout.flush()

    if resolution is None:
        resolution = vacuum_wavelength / 100
        sys.stdout.write('Setting lookup resolution to %f\n'%resolution)
        sys.stdout.flush()

    l_max = max([particle.l_max for particle in particle_list])
    m_max = max([particle.m_max for particle in particle_list])
    blocksize = fldex.blocksize(l_max, m_max)
    
    particle_x_array = np.array([particle.position[0] for particle in particle_list])
    particle_y_array = np.array([particle.position[1] for particle in particle_list])
    particle_z_array = np.array([particle.position[2] for particle in particle_list])
    particle_rho_array = np.sqrt((particle_x_array[:, None] - particle_x_array[None, :]) ** 2 
                                 + (particle_y_array[:, None] - particle_y_array[None, :]) ** 2)
    
    dz_min = particle_z_array.min() - particle_z_array.max()
    dz_max = particle_z_array.max() - particle_z_array.min()
    sz_min = 2 * particle_z_array.min()
    sz_max = 2 * particle_z_array.max()
    
    rho_array = np.arange(- 3 * resolution, particle_rho_array.max() + 3 * resolution, resolution)
    sz_array = np.arange(sz_min - 3 * resolution, sz_max + 3 * resolution, resolution)
    dz_array = np.arange(dz_min - 3 * resolution, dz_max + 3 * resolution, resolution)
    
    len_rho = len(rho_array)
    len_sz = len(sz_array)
    len_dz = len(dz_array)
    assert len_sz == len_dz
    
    i_s = layer_system.layer_number(particle_list[0].position[2])
    k_is = layer_system.wavenumber(i_s, vacuum_wavelength)
    z_is = layer_system.reference_z(i_s)
    
    # direct -----------------------------------------------------------------------------------------------------------
    w = np.zeros((len_rho, len_dz, blocksize, blocksize), dtype=np.complex64)
    sys.stdout.write('Lookup table memory footprint: ' + size_format(2 * w.nbytes) + '\n')
    sys.stdout.flush()

    r_array = np.sqrt(dz_array[None, :]**2 + rho_array[:, None]**2)
    r_array[r_array==0] = 1e-20
    ct = dz_array[None, :] / r_array
    st = rho_array[:, None] / r_array
    legendre, _, _ = sf.legendre_normalized(ct, st, 2 * l_max)
    
    bessel_h = []
    for dm in tqdm(range(2 * l_max + 1), desc='Spherical Hankel lookup   ', file=sys.stdout,
                   bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}'):
        bessel_h.append(sf.spherical_hankel(dm, k_is * r_array))
    
    pbar = tqdm(total=blocksize**2, 
                desc='Direct coupling           ', 
                file=sys.stdout,
                bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}')
            
    for m1 in range(-m_max, m_max+1):
        for m2 in range(-m_max, m_max+1):
            for l1 in range(max(1, abs(m1)), l_max + 1):
                for l2 in range(max(1, abs(m2)), l_max + 1):
                    A = np.zeros((len_rho, len_dz), dtype=complex)
                    B = np.zeros((len_rho, len_dz), dtype=complex)
                    for ld in range(max(abs(l1 - l2), abs(m1 - m2)), l1 + l2 + 1):  # if ld<abs(m1-m2) then P=0
                        a5, b5 = vwf.ab5_coefficients(l2, m2, l1, m1, ld)    # remember that w = A.T
                        A += a5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]   # remember that w = A.T
                        B += b5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]   # remember that w = A.T
                    for tau1 in range(2):
                        n1 = fldex.multi_to_single_index(tau1, l1, m1, l_max, m_max)
                        for tau2 in range(2):
                            n2 = fldex.multi_to_single_index(tau2, l2, m2, l_max, m_max)
                            if tau1 == tau2:
                                w[:, :, n1, n2] = A
                            else:
                                w[:, :, n1, n2] = B
                            pbar.update()
    pbar.close()

    # switch off direct coupling contribution near rho=0:
    w[rho_array < particle_rho_array[~np.eye(particle_rho_array.shape[0],dtype=bool)].min() / 2, :, :, :] = 0  

    # layer mediated ---------------------------------------------------------------------------------------------------
    sys.stdout.write('Layer mediated coupling   : ...')
    sys.stdout.flush()
    if type(k_parallel) == str and k_parallel == 'default':
        k_parallel = coord.default_k_parallel
    kz_is = coord.k_z(k_parallel=k_parallel, k=k_is)
    len_kp = len(k_parallel)

    # phase factors
    epljksz = np.exp(1j * kz_is[None, :] * (sz_array[:, None] - 2 * z_is))  # z, k
    emnjksz = np.exp(- 1j * kz_is[None, :] * (sz_array[:, None] - 2 * z_is))
    epljkdz = np.exp(1j * kz_is[None, :] * dz_array[:, None])
    emnjkdz = np.exp(- 1j * kz_is[None, :] * dz_array[:, None])
       
    # layer response
    L = np.zeros((2, 2, 2, len_kp), dtype=complex)  # pol, pl/mn1, pl/mn2, kp
    for pol in range(2):
        L[pol, :, :, :] = lay.layersystem_response_matrix(pol, layer_system.thicknesses,
                                                          layer_system.refractive_indices, k_parallel,
                                                          coord.angular_frequency(vacuum_wavelength), i_s, i_s)
   
    # transformation coefficients
    B_dag = np.zeros((2, 2, blocksize, len_kp), dtype=complex)  # pol, pl/mn, n, kp
    B = np.zeros((2, 2, blocksize, len_kp), dtype=complex)  # pol, pl/mn, n, kp
    ct_k = kz_is / k_is
    st_k = k_parallel / k_is
    _, pilm_pl, taulm_pl = sf.legendre_normalized(ct_k, st_k, l_max)
    _, pilm_mn, taulm_mn = sf.legendre_normalized(-ct_k, st_k, l_max)
    
    m_list = [None for i in range(blocksize)]
    for tau in range(2):
        for m in range(-m_max, m_max + 1):
            for l in range(max(1, abs(m)), l_max + 1):
                n = fldex.multi_to_single_index(tau, l, m, l_max, m_max)
                m_list[n] = m
                for pol in range(2):
                    B_dag[pol, 0, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl,
                                                                                    taulm_list=taulm_pl, dagger=True)
                    B_dag[pol, 1, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn,
                                                                                    taulm_list=taulm_mn, dagger=True)
                    B[pol, 0, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl,
                                                                             taulm_list=taulm_pl, dagger=False)
                    B[pol, 1, n, :] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn,
                                                                             taulm_list=taulm_mn, dagger=False)
    
    # pairs of (n1, n2), listed by abs(m1-m2)
    n1n2_combinations = [[] for dm in range(2*m_max+1)]
    for n1 in range(blocksize):
        m1 = m_list[n1]
        for n2 in range(blocksize):
            m2 = m_list[n2]
            n1n2_combinations[abs(m1-m2)].append((n1,n2))
                   
    wr_pl = np.zeros((len_rho, len_dz, blocksize, blocksize), dtype=np.complex64)
    wr_mn = np.zeros((len_rho, len_dz, blocksize, blocksize), dtype=np.complex64)
    
    dkp = np.diff(k_parallel)
    if cu.use_gpu:
        re_dkp_d = gpuarray.to_gpu(np.float32(dkp.real))
        im_dkp_d = gpuarray.to_gpu(np.float32(dkp.imag))
        kernel_source_code = cu.volume_lookup_assembly_code %(blocksize, len_rho, len_sz, len_kp)
        helper_function = SourceModule(kernel_source_code).get_function("helper")
        cuda_blocksize = 128
        cuda_gridsize = (len_rho * len_sz + cuda_blocksize - 1) // cuda_blocksize
    
        re_dwr_d = gpuarray.to_gpu(np.zeros((len_rho, len_sz), dtype=np.float32))
        im_dwr_d = gpuarray.to_gpu(np.zeros((len_rho, len_sz), dtype=np.float32))
    
    pbar = tqdm(total=blocksize**2, 
                desc='Layer mediated coupling   ', 
                file=sys.stdout,
                bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}')
            
    for dm in range(2*m_max+1):
        bessel = scipy.special.jv(dm, (k_parallel[None,:]*rho_array[:,None]))
        besjac = bessel * (k_parallel / (kz_is * k_is))[None,:]

        for n1n2 in n1n2_combinations[dm]:
            n1 = n1n2[0]
            m1 = m_list[n1]
            n2 = n1n2[1]
            m2 = m_list[n2]
    
            belbee_pl = np.zeros((len_dz, len_kp), dtype=complex)
            belbee_mn = np.zeros((len_dz, len_kp), dtype=complex)
            for pol in range(2):
                belbee_pl += ((L[pol, 0, 1, :] * B_dag[pol, 0, n1, :] * B[pol, 1, n2, :])[None, :] * epljksz
                              + (L[pol, 1, 0, :] * B_dag[pol, 1, n1, :] * B[pol, 0, n2, :])[None, :] * emnjksz)
                belbee_mn += ((L[pol, 0, 0, :] * B_dag[pol, 0, n1, :] * B[pol, 0, n2, :])[None, :] * epljkdz
                              + (L[pol, 1, 1, :] * B_dag[pol, 1, n1, :] * B[pol, 1, n2, :])[None, :] * emnjkdz)
            
            if cu.use_gpu:
                re_belbee_pl_d = gpuarray.to_gpu(np.float32(belbee_pl[None, :, :].real))
                im_belbee_pl_d = gpuarray.to_gpu(np.float32(belbee_pl[None, :, :].imag))
                re_belbee_mn_d = gpuarray.to_gpu(np.float32(belbee_mn[None, :, :].real))
                im_belbee_mn_d = gpuarray.to_gpu(np.float32(belbee_mn[None, :, :].imag))
                
                re_besjac_d = gpuarray.to_gpu(np.float32(besjac[:, None, :].real))
                im_besjac_d = gpuarray.to_gpu(np.float32(besjac[:, None, :].imag))
            
                helper_function(re_besjac_d.gpudata, im_besjac_d.gpudata, re_belbee_pl_d.gpudata,
                                im_belbee_pl_d.gpudata, re_dkp_d.gpudata, im_dkp_d.gpudata, re_dwr_d.gpudata, 
                                im_dwr_d.gpudata, block=(cuda_blocksize, 1, 1), grid=(cuda_gridsize, 1))
                wr_pl[:, :, n1, n2] = 4 * (1j)**abs(m2 - m1) * (re_dwr_d.get() + 1j * im_dwr_d.get()) 
                
                helper_function(re_besjac_d.gpudata, im_besjac_d.gpudata, re_belbee_mn_d.gpudata, 
                                im_belbee_mn_d.gpudata, re_dkp_d.gpudata, im_dkp_d.gpudata, re_dwr_d.gpudata, 
                                im_dwr_d.gpudata, block=(cuda_blocksize, 1, 1), grid=(cuda_gridsize, 1))
                wr_mn[:, :, n1, n2] = 4 * (1j)**abs(m2 - m1) * (re_dwr_d.get() + 1j * im_dwr_d.get()) 
            else:
                integrand = besjac[:, None, :] * belbee_pl[None, :, :] 
                wr_pl[:, :, n1, n2] = 2 * (1j)**abs(m2 - m1) * ((integrand[:, :, :-1] + integrand[:, :, 1:]) 
                                                                * dkp[None, None, :]).sum(axis=-1)   # trapezoidal rule
                
                integrand = besjac[:, None, :] * belbee_mn[None, :, :]
                wr_mn[:, :, n1, n2] = 2 * (1j)**abs(m2 - m1) * ((integrand[:, :, :-1] + integrand[:, :, 1:])
                                                                * dkp[None, None, :]).sum(axis=-1)
            pbar.update()
    pbar.close()
    
    return wr_pl, w + wr_mn, rho_array, sz_array, dz_array


def radial_coupling_lookup_table(vacuum_wavelength, particle_list, layer_system, k_parallel='default', resolution=None):
    """Prepare Sommerfeld integral lookup table to allow for a fast calculation of the coupling matrix by interpolation.
    This function is called when all particles are on the same z-position.
    
    Args:
        vacuum_wavelength (float):  Vacuum wavelength in length units
        particle_list (list):       List of particle objects
        layer_system (smuthi.layers.LayerSystem):    Stratified medium
        k_parallel (numpy.ndarray or str):           In-plane wavenumber for Sommerfeld integrals.
                                                     If 'default', smuthi.coordinates.default_k_parallel
        resolution (float): Spatial resolution of lookup table in length units. (default: vacuum_wavelength / 100)       
                            Smaller means more accurate but higher memory footprint 
                            
    Returns:
        (tuple) tuple containing:
        
            lookup_table (ndarray):  Coupling lookup, indices are [rho, n1, n2].
            rho_array (ndarray):     Values for the radial distance considered for the lookup (starting from negative 
                                     numbers to allow for simpler cubic interpolation without distinction of cases 
                                     at rho=0)
    """

    sys.stdout.write('Prepare radial particle coupling lookup:\n')
    sys.stdout.flush()

    if resolution is None:
        resolution = vacuum_wavelength / 100
        sys.stdout.write('Setting lookup resolution to %f\n'%resolution)
        sys.stdout.flush()
   
    l_max = max([particle.l_max for particle in particle_list])
    m_max = max([particle.m_max for particle in particle_list])
    blocksize = fldex.blocksize(l_max, m_max)
    
    x_array = np.array([particle.position[0] for particle in particle_list])
    y_array = np.array([particle.position[1] for particle in particle_list])
    rho_array = np.sqrt((x_array[:, None] - x_array[None, :]) ** 2 + (y_array[:, None] - y_array[None, :]) ** 2)

    radial_distance_array = np.arange(- 3 * resolution, rho_array.max() + 3 * resolution, resolution)
    
    z = particle_list[0].position[2]
    i_s = layer_system.layer_number(z)
    k_is = layer_system.wavenumber(i_s, vacuum_wavelength)
    dz = z - layer_system.reference_z(i_s)
    
    len_rho = len(radial_distance_array)
        
    # direct -----------------------------------------------------------------------------------------------------------
    w = np.zeros((len_rho, blocksize, blocksize), dtype=np.complex64)
    sys.stdout.write('Memory footprint: ' + size_format(w.nbytes) + '\n')
    sys.stdout.flush()

    ct = np.array([0.0])
    st = np.array([1.0])
    bessel_h = []
    for n in range(2* l_max + 1):
        bessel_h.append(sf.spherical_hankel(n, k_is * radial_distance_array))
        bessel_h[-1][radial_distance_array <= 0] = np.nan
        
    legendre, _, _ = sf.legendre_normalized(ct, st, 2 * l_max)

    pbar = tqdm(total=blocksize**2, 
                desc='Direct coupling           ', 
                file=sys.stdout,
                bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}')

    for m1 in range(-m_max, m_max+1):
        for m2 in range(-m_max, m_max+1):
            for l1 in range(max(1, abs(m1)), l_max + 1):
                for l2 in range(max(1, abs(m2)), l_max + 1):
                    A = np.zeros(len_rho, dtype=complex)
                    B = np.zeros(len_rho, dtype=complex)
                    for ld in range(max(abs(l1 - l2), abs(m1 - m2)), l1 + l2 + 1):  # if ld<abs(m1-m2) then P=0
                        a5, b5 = vwf.ab5_coefficients(l2, m2, l1, m1, ld)
                        A += a5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                        B += b5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                    for tau1 in range(2):
                        n1 = fldex.multi_to_single_index(tau1, l1, m1, l_max, m_max)
                        for tau2 in range(2):
                            n2 = fldex.multi_to_single_index(tau2, l2, m2, l_max, m_max)
                            if tau1 == tau2:
                                w[:, n1, n2] = A  # remember that w = A.T
                            else:
                                w[:, n1, n2] = B
                            pbar.update()
    pbar.close()
    close_to_zero = radial_distance_array < rho_array[~np.eye(rho_array.shape[0],dtype=bool)].min() / 2 
    w[close_to_zero, :, :] = 0  # switch off direct coupling contribution near rho=0

    # layer mediated ---------------------------------------------------------------------------------------------------
    sys.stdout.write('Layer mediated coupling   : ...')
    sys.stdout.flush()
    
    if type(k_parallel) == str and k_parallel == 'default':
        k_parallel = coord.default_k_parallel
    kz_is = coord.k_z(k_parallel=k_parallel, k=k_is)
    len_kp = len(k_parallel)

    # phase factors
    epl2jkz = np.exp(2j * kz_is * dz)
    emn2jkz = np.exp(-2j * kz_is * dz)
       
    # layer response
    L = np.zeros((2,2,2,len_kp), dtype=complex)  # pol, pl/mn1, pl/mn2, kp
    for pol in range(2):
        L[pol,:,:,:] = lay.layersystem_response_matrix(pol, layer_system.thicknesses,
                                                       layer_system.refractive_indices, k_parallel,
                                                       coord.angular_frequency(vacuum_wavelength), i_s, i_s)
   
    # transformation coefficients
    B_dag = np.zeros((2, 2, blocksize, len_kp), dtype=complex)  # pol, pl/mn, n, kp
    B = np.zeros((2, 2, blocksize, len_kp), dtype=complex)  # pol, pl/mn, n, kp
    ct = kz_is / k_is
    st = k_parallel / k_is
    _, pilm_pl, taulm_pl = sf.legendre_normalized(ct, st, l_max)
    _, pilm_mn, taulm_mn = sf.legendre_normalized(-ct, st, l_max)

    m_list = [None for n in range(blocksize)]
    for tau in range(2):
        for m in range(-m_max, m_max + 1):
            for l in range(max(1, abs(m)), l_max + 1):
                n = fldex.multi_to_single_index(tau, l, m, l_max, m_max)
                m_list[n] = m
                for pol in range(2):
                    B_dag[pol,0,n,:] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl,
                                                                           taulm_list=taulm_pl, dagger=True)
                    B_dag[pol,1,n,:] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn,
                                                                           taulm_list=taulm_mn, dagger=True)
                    B[pol,0,n,:] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl,
                                                                       taulm_list=taulm_pl, dagger=False)
                    B[pol,1,n,:] = vwf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn,
                                                                       taulm_list=taulm_mn, dagger=False)
    
    # pairs of (n1, n2), listed by abs(m1-m2)
    n1n2_combinations = [[] for dm in range(2*m_max+1)]
    for n1 in range(blocksize):
        m1 = m_list[n1]
        for n2 in range(blocksize):
            m2 = m_list[n2]
            n1n2_combinations[abs(m1-m2)].append((n1,n2))
                   
    wr = np.zeros((len_rho, blocksize, blocksize), dtype=complex)
    
    dkp = np.diff(k_parallel)
    if cu.use_gpu:
        re_dkp_d = gpuarray.to_gpu(np.float32(dkp.real))
        im_dkp_d = gpuarray.to_gpu(np.float32(dkp.imag))
        kernel_source_code = cu.radial_lookup_assembly_code %(blocksize, len_rho, len_kp)
        helper_function = SourceModule(kernel_source_code).get_function("helper")
        cuda_blocksize = 128
        cuda_gridsize = (len_rho + cuda_blocksize - 1) // cuda_blocksize
    
        re_dwr_d = gpuarray.to_gpu(np.zeros(len_rho, dtype=np.float32))
        im_dwr_d = gpuarray.to_gpu(np.zeros(len_rho, dtype=np.float32))    
    
    n1n2_combinations = [[] for dm in range(2*m_max+1)]
    for n1 in range(blocksize):
        m1 = m_list[n1]
        for n2 in range(blocksize):
            m2 = m_list[n2]
            n1n2_combinations[abs(m1-m2)].append((n1,n2))
    
    pbar = tqdm(total=blocksize**2, 
                desc='Layer mediated coupling   ', 
                file=sys.stdout,
                bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}')
            
    for dm in range(2*m_max+1):
        bessel = scipy.special.jv(dm, (k_parallel[None,:]*radial_distance_array[:,None]))
        besjac = bessel * (k_parallel / (kz_is * k_is))[None,:]
        for n1n2 in n1n2_combinations[dm]:
            n1 = n1n2[0]
            m1 = m_list[n1]
            n2 = n1n2[1]
            m2 = m_list[n2]
            
            belbe = np.zeros(len_kp, dtype=complex)  # n1, n2, kp
            for pol in range(2):
                belbe += L[pol,0,0,:] * B_dag[pol,0,n1,:] * B[pol,0,n2,:]
                belbe += L[pol,1,0,:] * B_dag[pol,1,n1,:] * B[pol,0,n2,:] * emn2jkz
                belbe += L[pol,0,1,:] * B_dag[pol,0,n1,:] * B[pol,1,n2,:] * epl2jkz
                belbe += L[pol,1,1,:] * B_dag[pol,1,n1,:] * B[pol,1,n2,:]
            
            if cu.use_gpu:
                re_belbe_d = gpuarray.to_gpu(np.float32(belbe[None, :].real))
                im_belbe_d = gpuarray.to_gpu(np.float32(belbe[None, :].imag))
                
                re_besjac_d = gpuarray.to_gpu(np.float32(besjac.real))
                im_besjac_d = gpuarray.to_gpu(np.float32(besjac.imag))
            
                helper_function(re_besjac_d.gpudata, im_besjac_d.gpudata, 
                                re_belbe_d.gpudata, im_belbe_d.gpudata, 
                                re_dkp_d.gpudata, im_dkp_d.gpudata, 
                                re_dwr_d.gpudata, im_dwr_d.gpudata, 
                                block=(cuda_blocksize, 1, 1), grid=(cuda_gridsize, 1))
                
                wr[:,n1,n2] = 4 * (1j)**abs(m2-m1) * (re_dwr_d.get() + 1j*im_dwr_d.get()) 

            else:
                integrand = besjac * belbe[None, :]  # rho, kp 
                wr[:,n1,n2] = 2 * (1j)**abs(m2-m1) * ((integrand[:,:-1] + integrand[:,1:]) 
                                                            * dkp[None,:]).sum(axis=-1)  # trapezoidal rule
            pbar.update()
    pbar.close()
    
    return w + wr, radial_distance_array


def size_format(b):
    if b < 1000:
              return '%i' % b + 'B'
    elif 1000 <= b < 1000000:
        return '%.1f' % float(b/1000) + 'KB'
    elif 1000000 <= b < 1000000000:
        return '%.1f' % float(b/1000000) + 'MB'
    elif 1000000000 <= b < 1000000000000:
        return '%.1f' % float(b/1000000000) + 'GB'
    elif 1000000000000 <= b:
        return '%.1f' % float(b/1000000000000) + 'TB'


def spheroids_closest_points(ab_halfaxis1, c_halfaxis1, center1, orientation1, ab_halfaxis2, c_halfaxis2, center2, 
                             orientation2):
    """ Computation of the two closest points of two adjacent spheroids.
    For details, see: Stephen B. Pope, Algorithms for Ellipsoids, Sibley School of Mechanical & Aerospace Engineering, 
    Cornell University, Ithaca, New York, February 2008
    
    Args:
        ab_halfaxis1 (float):        Half axis orthogonal to symmetry axis of spheroid 1
        c_halfaxis1 (float):         Half axis parallel to symmetry axis of spheroid 1
        center1 (numpy.array):       Center coordinates of spheroid 1
        orientation1 (numpy.array):  Orientation angles of spheroid 1
        ab_halfaxis2 (float):        Half axis orthogonal to symmetry axis of spheroid 2
        c_halfaxis2 (float):         Half axis parallel to symmetry axis of spheroid 2
        center2 (numpy.array):       Center coordinates of spheroid 2
        orientation2 (numpy.array):  Orientation angles of spheroid 2
        
    Retruns:
        Tuple containing:
          - closest point on first particle (numpy.array)
          - closest point on second particle (numpy.array)
          - first rotation Euler angle alpha (float)
          - second rotation Euler angle beta (float)
    """
    
    def rotation_matrix(ang):
        rot_mat = (np.array([[np.cos(ang[0]) * np.cos(ang[1]), -np.sin(ang[0]), np.cos(ang[0]) * np.sin(ang[1])],
                             [np.sin(ang[0]) * np.cos(ang[1]), np.cos(ang[0]), np.sin(ang[0]) * np.sin(ang[1])],
                             [-np.sin(ang[1]), 0, np.cos(ang[1])]]))
        return rot_mat
    
    rot_matrix_1 = rotation_matrix(orientation1)
    rot_matrix_2 = rotation_matrix(orientation2)
        
    a1, a2 = ab_halfaxis1, ab_halfaxis2
    c1, c2 = c_halfaxis1, c_halfaxis2
    ctr1, ctr2 = np.array(center1), np.array(center2)
    
    eigenvalue_matrix_1 = np.array([[1 / a1 ** 2, 0, 0], [0, 1 / a1 ** 2, 0], [0, 0, 1 / c1 ** 2]])
    eigenvalue_matrix_2 = np.array([[1 / a2 ** 2, 0, 0], [0, 1 / a2 ** 2, 0], [0, 0, 1 / c2 ** 2]])
    
    E1 = np.dot(rot_matrix_1, np.dot(eigenvalue_matrix_1, np.transpose(rot_matrix_1)))
    E2 = np.dot(rot_matrix_2, np.dot(eigenvalue_matrix_2, np.transpose(rot_matrix_2)))
    S = np.matrix.getH(np.linalg.cholesky(E1))
    
    # transformation of spheroid E1 into the unit-sphere with its center at origin / same transformation on E2
    # E1_prime = np.dot(np.transpose(np.linalg.inv(S)), np.dot(E1, np.linalg.inv(S)))
    # ctr1_prime = ctr1 - ctr1 
    E2_prime = np.dot(np.transpose(np.linalg.inv(S)), np.dot(E2, np.linalg.inv(S)))
    ctr2_prime = -(np.dot(S, (ctr1 - ctr2)))  
    E2_prime_L = np.linalg.cholesky(E2_prime)
        
    H = np.dot(np.linalg.inv(E2_prime_L), np.transpose(np.linalg.inv(E2_prime_L)))
    p = np.array([0, 0, 0])
    f = np.dot(np.transpose(ctr2_prime - p), np.transpose(np.linalg.inv(E2_prime_L)))
    
    def minimization_fun(y_vec):
        fun = 0.5 * np.dot(np.dot(np.transpose(y_vec), H), y_vec) + np.dot(f, y_vec)
        return fun
    def constraint_fun(x):
        eq_constraint = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5 - 1
        return eq_constraint
    
    bnds = ((-1, 1), (-1, 1), (-1, 1))
    length_constraints = {'type' : 'eq', 'fun' : constraint_fun}
    
    flag = False
    while flag == False:
        x0 = -1 + np.dot((1 + 1), np.random.rand(3))
        optimization_result = scipy.optimize.minimize(minimization_fun, x0, method='SLSQP', bounds=bnds,
                                                      constraints=length_constraints, tol=None, callback=None, options=None)
        x_vec = np.transpose(np.dot(np.transpose(np.linalg.inv(E2_prime_L)), optimization_result['x'])
                             + np.transpose(ctr2_prime))
        if optimization_result['success'] == True:
            if np.linalg.norm(x_vec) <= 1:
                raise ValueError("particle error: particles intersect")
            elif np.linalg.norm(x_vec) < np.linalg.norm(ctr2_prime):
                flag = True
            else:
                print('wrong minimum ...')
        else:
            print('No minimum found ...')

    p2_prime = x_vec
    p2 = np.dot(np.linalg.inv(S), p2_prime) + ctr1
    
    E1_L = np.linalg.cholesky(E1)
    H = np.dot(np.linalg.inv(E1_L), np.transpose(np.linalg.inv(E1_L)))
    p = p2
    f = np.dot(np.transpose(ctr1 - p), np.transpose(np.linalg.inv(E1_L)))
       
    flag = False
    while flag == False:
        x0 = -1 + np.dot((1 + 1), np.random.rand(3))
        optimization_result2 = scipy.optimize.minimize(minimization_fun, x0, method='SLSQP', bounds=bnds,
                                                      constraints=length_constraints, tol=None, callback=None, options=None)
        p1 = np.transpose(np.dot(np.transpose(np.linalg.inv(E1_L)), optimization_result2['x']) + np.transpose(ctr1))
        if optimization_result2['success'] == True:
            if np.linalg.norm(p1 - p) < np.linalg.norm(ctr1 - p):
                flag = True
            else:
                print('wrong minimum ...')
        else:
            print('No minimum found ...')
    
    p1p2 = p2 - p1
    azimuth = np.arctan2(p1p2[1], p1p2[0])
    elevation = np.arctan2(p1p2[2], (p1p2[0] ** 2 + p1p2[1] ** 2) ** 0.5)

    if p1p2[2] < 0:
        beta = (np.pi / 2) + elevation
    else:
        beta = (-np.pi / 2) + elevation
    alpha = -azimuth
              
    return p1, p2, alpha, beta


def direct_coupling_block_pvwf_mediated(vacuum_wavelength, receiving_particle, emitting_particle, layer_system, 
                                        k_parallel):
    """Direct particle coupling matrix :math:`W` for two particles (via plane vector wave functions).
    For details, see: 
    Dominik Theobald et al., Phys. Rev. A 96, 033822, DOI: 10.1103/PhysRevA.96.033822 or arXiv:1708.04808 


    Args:
        vacuum_wavelength (float):                          Vacuum wavelength :math:`\lambda` (length unit)
        receiving_particle (smuthi.particles.Particle):     Particle that receives the scattered field
        emitting_particle (smuthi.particles.Particle):      Particle that emits the scattered field
        layer_system (smuthi.layers.LayerSystem):           Stratified medium in which the coupling takes place
        k_parallel (numpy.array):                           In-plane wavenumber for plane wave expansion

    Returns:
        Direct coupling matrix block (numpy array).
    """    
    if type(receiving_particle).__name__ != 'Spheroid' or type(emitting_particle).__name__ != 'Spheroid':
        raise NotImplementedError('plane wave coupling currently implemented only for spheroids')
    
    lmax1 = receiving_particle.l_max
    mmax1 = receiving_particle.m_max
    assert lmax1 == mmax1, 'PVWF coupling requires lmax == mmax for each particle.'
    lmax2 = emitting_particle.l_max
    mmax2 = emitting_particle.m_max
    assert lmax2 == mmax2, 'PVWF coupling requires lmax == mmax for each particle.'
    lmax = max([lmax1, lmax2])
    m_max = max([mmax1, mmax2]) 
    blocksize1 = fldex.blocksize(lmax1, mmax1)
    blocksize2 = fldex.blocksize(lmax2, mmax2)
    
    n_medium = layer_system.refractive_indices[layer_system.layer_number(receiving_particle.position[2])]
      
    # finding the orientation of a plane separating the spheroids
    _, _, alpha, beta = spheroids_closest_points(
        emitting_particle.semi_axis_a, emitting_particle.semi_axis_c, emitting_particle.position, 
        emitting_particle.euler_angles, receiving_particle.semi_axis_a, receiving_particle.semi_axis_c,
        receiving_particle.position, receiving_particle.euler_angles)
    
    # positions
    r1 = np.array(receiving_particle.position)
    r2 = np.array(emitting_particle.position)
    r21_lab = r1 - r2  # laboratory coordinate system
    
    # distance vector in rotated coordinate system
    r21_rot = np.dot(np.dot([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [- np.sin(beta), 0, np.cos(beta)]],
                           [[np.cos(alpha), - np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]]), 
                    r21_lab)
    rho21 = (r21_rot[0] ** 2 + r21_rot[1] ** 2) ** 0.5 
    phi21 = np.arctan2(r21_rot[1], r21_rot[0])
    z21 = r21_rot[2]
    
    # wavenumbers
    omega = coord.angular_frequency(vacuum_wavelength)
    k = omega * n_medium
    kz = coord.k_z(k_parallel=k_parallel, vacuum_wavelength=vacuum_wavelength, refractive_index=n_medium)
    if z21 < 0:
        kz_var = -kz
    else:
        kz_var = kz
        
    # Bessel lookup 
    bessel_list = []
    for dm in range(mmax1 + mmax2 + 1):
        bessel_list.append(scipy.special.jn(dm, k_parallel * rho21))
    
    # legendre function lookups
    ct = kz_var / k
    st = k_parallel / k
    _, pilm_list, taulm_list = sf.legendre_normalized(ct, st, lmax)
    
    # initialize result
    w = np.zeros((blocksize1, blocksize2), dtype=complex)

    # prefactor
    const_arr = k_parallel / (kz * k) * np.exp(1j * (kz_var * z21))
                        
    for m1 in range(-mmax1, mmax1 + 1):
        for m2 in range(-mmax2, mmax2 + 1):
            jmm_eimphi_bessel = 4 * 1j ** abs(m2 - m1) * np.exp(1j * phi21 * (m2 - m1)) * bessel_list[abs(m2 - m1)]
            prefactor = const_arr * jmm_eimphi_bessel
            for l1 in range(max(1, abs(m1)), lmax1 + 1):
                for l2 in range(max(1, abs(m2)), lmax2 + 1):
                    for tau1 in range(2):
                        n1 = fldex.multi_to_single_index(tau1, l1, m1, lmax1, mmax1)
                        for tau2 in range(2):
                            n2 = fldex.multi_to_single_index(tau2, l2, m2, lmax2, mmax2)
                            for pol in range(2):
                                B_dag = vwf.transformation_coefficients_vwf(tau1, l1, m1, pol, pilm_list=pilm_list, 
                                                                        taulm_list=taulm_list, dagger=True)
                                B = vwf.transformation_coefficients_vwf(tau2, l2, m2, pol, pilm_list=pilm_list,
                                                                            taulm_list=taulm_list, dagger=False)
                                integrand = prefactor * B * B_dag
                                w[n1, n2] += np.trapz(integrand, k_parallel) 
                                
    rot_mat_1 = fldex.block_rotation_matrix_D_svwf(lmax1, mmax1, 0, beta, alpha)
    rot_mat_2 = fldex.block_rotation_matrix_D_svwf(lmax2, mmax2, -alpha, -beta, 0)
    
    return np.dot(np.dot(np.transpose(rot_mat_1), w), np.transpose(rot_mat_2))
