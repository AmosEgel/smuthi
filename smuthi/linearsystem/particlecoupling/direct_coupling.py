"""This module contains functions to compute the direct (i.e., not layer 
mediated) particle coupling coefficients."""

import numpy as np
import smuthi.fields as flds
import smuthi.fields.transformations as trf
import smuthi.utility.math as sf
import scipy.optimize
import scipy.special


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
    omega = flds.angular_frequency(vacuum_wavelength)

    # index specs
    lmax1 = receiving_particle.l_max
    mmax1 = receiving_particle.m_max
    lmax2 = emitting_particle.l_max
    mmax2 = emitting_particle.m_max
    blocksize1 = flds.blocksize(lmax1, mmax1)
    blocksize2 = flds.blocksize(lmax2, mmax2)

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
                            a5, b5 = trf.ab5_coefficients(l2, m2, l1, m1, ld)
                            A += a5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                            B += b5 * bessel_h[ld] * legendre[ld][abs(m1 - m2)]
                        A, B = eimph * A, eimph * B
                        for tau1 in range(2):
                            n1 = flds.multi_to_single_index(tau1, l1, m1, lmax1, mmax1)
                            for tau2 in range(2):
                                n2 = flds.multi_to_single_index(tau2, l2, m2, lmax2, mmax2)
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
    blocksizes = [flds.blocksize(particle.l_max, particle.m_max)
                  for particle in particle_list]

    # initialize result
    w = np.zeros((sum(blocksizes), sum(blocksizes)), dtype=complex)

    for s1, particle1 in enumerate(particle_list):
        idx1 = np.array(range(sum(blocksizes[:s1]), sum(blocksizes[:s1+1])))
        for s2, particle2 in enumerate(particle_list):
            idx2 = range(sum(blocksizes[:s2]), sum(blocksizes[:s2+1]))
            w[idx1[:, None], idx2] = direct_coupling_block(vacuum_wavelength, particle1, particle2, layer_system)

    return w


###############################################################################
#                  PVWF coupling - experimental!                              #                               
###############################################################################
"""The following code section contains functions to compute the particle 
coupling via a PVWF expansion. This allows in principle to treat particles
with intersecting circumscribing spheres, see 
Theobald et al.: "Plane-wave coupling formalism for T-matrix simulations of 
light scattering by nonspherical particles", Phys Rev A, 2018"""

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
    while not flag:
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
    while not flag:
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
    blocksize1 = flds.blocksize(lmax1, mmax1)
    blocksize2 = flds.blocksize(lmax2, mmax2)
    
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
    omega = flds.angular_frequency(vacuum_wavelength)
    k = omega * n_medium
    kz = flds.k_z(k_parallel=k_parallel, vacuum_wavelength=vacuum_wavelength, refractive_index=n_medium)
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
                        n1 = flds.multi_to_single_index(tau1, l1, m1, lmax1, mmax1)
                        for tau2 in range(2):
                            n2 = flds.multi_to_single_index(tau2, l2, m2, lmax2, mmax2)
                            for pol in range(2):
                                B_dag = trf.transformation_coefficients_vwf(tau1, l1, m1, pol, pilm_list=pilm_list,
                                                                        taulm_list=taulm_list, dagger=True)
                                B = trf.transformation_coefficients_vwf(tau2, l2, m2, pol, pilm_list=pilm_list,
                                                                            taulm_list=taulm_list, dagger=False)
                                integrand = prefactor * B * B_dag
                                w[n1, n2] += np.trapz(integrand, k_parallel) 
                                
    rot_mat_1 = trf.block_rotation_matrix_D_svwf(lmax1, mmax1, 0, beta, alpha)
    rot_mat_2 = trf.block_rotation_matrix_D_svwf(lmax2, mmax2, -alpha, -beta, 0)
    
    return np.dot(np.dot(np.transpose(rot_mat_1), w), np.transpose(rot_mat_2))
