"""This module contains functionality to prepare lookups for the particle 
coupling coefficients, which allows to efficiently treat large numbers of 
particles. """

import sys
from tqdm import tqdm
import numpy as np
import scipy.special
import smuthi.fields
import smuthi.utility.math as sf
import smuthi.utility.cuda as cu
import smuthi.fields.transformations as trf
import smuthi.layers as lay
import smuthi.linearsystem.particlecoupling.prepare_lookup_cuda as cusrc

def volumetric_coupling_lookup_table(vacuum_wavelength, particle_list, layer_system, k_parallel='default', 
                                     resolution=None):
    """Prepare Sommerfeld integral lookup table to allow for a fast calculation of the coupling matrix by interpolation.
    This function is called when not all particles are on the same z-position.
    
    Args:
        vacuum_wavelength (float):  Vacuum wavelength in length units
        particle_list (list):       List of particle objects
        layer_system (smuthi.layers.LayerSystem):    Stratified medium
        k_parallel (numpy.ndarray or str):           In-plane wavenumber for Sommerfeld integrals.
                                                     If 'default', smuthi.fields.default_Sommerfeld_k_parallel_array
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
    blocksize = smuthi.fields.blocksize(l_max, m_max)
    
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
                        a5, b5 = trf.ab5_coefficients(l2, m2, l1, m1, ld)    # remember that w = A.T
                        A += a5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]   # remember that w = A.T
                        B += b5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]   # remember that w = A.T
                    for tau1 in range(2):
                        n1 = smuthi.fields.multi_to_single_index(tau1, l1, m1, l_max, m_max)
                        for tau2 in range(2):
                            n2 = smuthi.fields.multi_to_single_index(tau2, l2, m2, l_max, m_max)
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
        k_parallel = smuthi.fields.default_Sommerfeld_k_parallel_array
    kz_is = smuthi.fields.k_z(k_parallel=k_parallel, k=k_is)
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
                                                          smuthi.fields.angular_frequency(vacuum_wavelength), i_s, i_s)
   
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
                n = smuthi.fields.multi_to_single_index(tau, l, m, l_max, m_max)
                m_list[n] = m
                for pol in range(2):
                    B_dag[pol, 0, n, :] = trf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl,
                                                                                    taulm_list=taulm_pl, dagger=True)
                    B_dag[pol, 1, n, :] = trf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn,
                                                                                    taulm_list=taulm_mn, dagger=True)
                    B[pol, 0, n, :] = trf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl,
                                                                             taulm_list=taulm_pl, dagger=False)
                    B[pol, 1, n, :] = trf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn,
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
        re_dkp_d = cu.gpuarray.to_gpu(np.float32(dkp.real))
        im_dkp_d = cu.gpuarray.to_gpu(np.float32(dkp.imag))
        kernel_source_code = cusrc.volume_lookup_assembly_code %(blocksize, len_rho, len_sz, len_kp)
        helper_function = cu.SourceModule(kernel_source_code).get_function("helper")
        cuda_blocksize = 128
        cuda_gridsize = (len_rho * len_sz + cuda_blocksize - 1) // cuda_blocksize

        re_dwr_d = cu.gpuarray.to_gpu(np.zeros((len_rho, len_sz), dtype=np.float32))
        im_dwr_d = cu.gpuarray.to_gpu(np.zeros((len_rho, len_sz), dtype=np.float32))

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
                re_belbee_pl_d = cu.gpuarray.to_gpu(np.float32(belbee_pl[None, :, :].real))
                im_belbee_pl_d = cu.gpuarray.to_gpu(np.float32(belbee_pl[None, :, :].imag))
                re_belbee_mn_d = cu.gpuarray.to_gpu(np.float32(belbee_mn[None, :, :].real))
                im_belbee_mn_d = cu.gpuarray.to_gpu(np.float32(belbee_mn[None, :, :].imag))

                re_besjac_d = cu.gpuarray.to_gpu(np.float32(besjac[:, None, :].real))
                im_besjac_d = cu.gpuarray.to_gpu(np.float32(besjac[:, None, :].imag))
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
                                                     If 'default', smuthi.fields.default_Sommerfeld_k_parallel_array
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
    blocksize = smuthi.fields.blocksize(l_max, m_max)
    
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
                        a5, b5 = trf.ab5_coefficients(l2, m2, l1, m1, ld)
                        A = A + a5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                        B = B + b5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                    for tau1 in range(2):
                        n1 = smuthi.fields.multi_to_single_index(tau1, l1, m1, l_max, m_max)
                        for tau2 in range(2):
                            n2 = smuthi.fields.multi_to_single_index(tau2, l2, m2, l_max, m_max)
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
        k_parallel = smuthi.fields.default_Sommerfeld_k_parallel_array

    kz_is = smuthi.fields.k_z(k_parallel=k_parallel, k=k_is)
    len_kp = len(k_parallel)

    # phase factors
    epl2jkz = np.exp(2j * kz_is * dz)
    emn2jkz = np.exp(-2j * kz_is * dz)
       
    # layer response
    L = np.zeros((2,2,2,len_kp), dtype=complex)  # pol, pl/mn1, pl/mn2, kp
    for pol in range(2):
        L[pol,:,:,:] = lay.layersystem_response_matrix(pol, layer_system.thicknesses,
                                                       layer_system.refractive_indices, k_parallel,
                                                       smuthi.fields.angular_frequency(vacuum_wavelength), i_s, i_s)
   
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
                n = smuthi.fields.multi_to_single_index(tau, l, m, l_max, m_max)
                m_list[n] = m
                for pol in range(2):
                    B_dag[pol,0,n,:] = trf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl,
                                                                           taulm_list=taulm_pl, dagger=True)
                    B_dag[pol,1,n,:] = trf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn,
                                                                           taulm_list=taulm_mn, dagger=True)
                    B[pol,0,n,:] = trf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_pl,
                                                                       taulm_list=taulm_pl, dagger=False)
                    B[pol,1,n,:] = trf.transformation_coefficients_vwf(tau, l, m, pol, pilm_list=pilm_mn,
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
        re_dkp_d = cu.gpuarray.to_gpu(np.float32(dkp.real))
        im_dkp_d = cu.gpuarray.to_gpu(np.float32(dkp.imag))
        kernel_source_code = cusrc.radial_lookup_assembly_code %(blocksize, len_rho, len_kp)
        helper_function = cu.SourceModule(kernel_source_code).get_function("helper")
        cuda_blocksize = 128
        cuda_gridsize = (len_rho + cuda_blocksize - 1) // cuda_blocksize

        re_dwr_d = cu.gpuarray.to_gpu(np.zeros(len_rho, dtype=np.float32))
        im_dwr_d = cu.gpuarray.to_gpu(np.zeros(len_rho, dtype=np.float32))
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
                re_belbe_d = cu.gpuarray.to_gpu(np.float32(belbe[None, :].real))
                im_belbe_d = cu.gpuarray.to_gpu(np.float32(belbe[None, :].imag))

                re_besjac_d = cu.gpuarray.to_gpu(np.float32(besjac.real))
                im_besjac_d = cu.gpuarray.to_gpu(np.float32(besjac.imag))
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

