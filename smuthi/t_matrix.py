# -*- coding: utf-8 -*-

import numpy as np
import smuthi.spherical_functions
import smuthi.index_conversion


def mie_coefficient(tau, l, k_medium, k_particle, radius):
    """Return the Mie coefficients of a sphere.

    Input:
    tau         integer: spherical polarization, 0 for spherical TE and 1 for spherical TM
    l           integer: l=1,... multipole degree (polar quantum number)
    k_medium    float or complex: wavenumber in surrounding medium (inverse length unit)
    k_particle  float or complex: wavenumber inside sphere (inverse length unit)
    radius      float: radius of sphere (length unit)
    """
    jlkr_medium = smuthi.spherical_functions.spherical_bessel(l, k_medium * radius)
    jlkr_particle = smuthi.spherical_functions.spherical_bessel(l, k_particle * radius)
    dxxj_medium = smuthi.spherical_functions.dx_xj(l, k_medium * radius)
    dxxj_particle = smuthi.spherical_functions.dx_xj(l, k_particle * radius)

    hlkr_medium = smuthi.spherical_functions.spherical_hankel(l, k_medium * radius)
    dxxh_medium = smuthi.spherical_functions.dx_xh(l, k_medium * radius)

    if tau == 0:
        q = (jlkr_medium * dxxj_particle - jlkr_particle * dxxj_medium) / (jlkr_particle * dxxh_medium - hlkr_medium *
                                                                           dxxj_particle)
    elif tau == 1:
        q = ((k_medium ** 2 * jlkr_medium * dxxj_particle - k_particle ** 2 * jlkr_particle * dxxj_medium) /
             (k_particle ** 2 * jlkr_particle * dxxh_medium - k_medium ** 2 * hlkr_medium * dxxj_particle))
    else:
        raise ValueError('tau must be 0 (spherical TE) or 1 (spherical TM)')

    return q

def t_matrix_sphere(k_medium, k_particle, radius, lmax=None, mmax=None, index_arrangement='stlm'):
    """Return the T-matrix of a spherical scattering object.

    Input:
    k_medium            float or complex: wavenumber in surrounding medium (inverse length unit)
    k_particle          float or complex: wavenumber inside sphere (inverse length unit)
    radius              float: radius of sphere (length unit)
    lmax:               truncation degree of SVWF expansion
    mmax:               (optional) truncation order of SVWF expansion, i.e., |m|<=mmax, default: mmax=lmax
    index_arrangement:  (optional) string to specify the order according to which the indices are arranged
                        See smuthi.index_conversion for explanation. Default: 'stlm'
    """
    if mmax is None:
        mmax = lmax
    blocksize = smuthi.index_conversion.block_size(lmax, mmax, 1, index_arrangement)
    t = np.zeros((blocksize, blocksize), dtype=complex)
    for tau in range(2):
        for m in range(-mmax, mmax + 1):
            for l in range(max(1, abs(m)), lmax+1):
                n = smuthi.index_conversion.multi2single(tau, l, m, lmax, mmax, index_arrangement=index_arrangement)
                t[n, n] = mie_coefficient(tau, l, k_medium, k_particle, radius)

    return t


def t_matrix(vacuum_wavelength, n_medium, particle, index_specs):
    """Return the T-matrix of a particle.

    NOT TESTED

    Input:
    vacuum_wavelength:  float (length units)
    n_medium:           float or complex: refractive index of surrounding medium
    particle:           Dictionary containing the particle parameters. See smuthi.particles for details.
    index_specs:        A dictionary with 'lmax', 'mmax' and 'index arrangement' (see index_conversion.py)
    """
    lmax = index_specs['lmax']
    mmax = index_specs['mmax']
    index_arrangement = index_specs['index arrangement']

    if particle['shape'] == 'sphere':
        k_medium = 2 * np.pi / vacuum_wavelength * n_medium
        k_particle = 2 * np.pi / vacuum_wavelength * particle['refractive index']
        radius = particle['radius']
        t = t_matrix_sphere(k_medium, k_particle, radius, lmax, mmax=mmax, index_arrangement=index_arrangement)
    else:
        raise ValueError('T-matrix for ' + particle['shape'] + ' currently not implemented.')

    return t


def rotate_t_matrix(t, euler_angles, index_specs):
    """Placeholder for a proper T-matrix rotation routine"""
    if euler_angles == [0, 0, 0]:
        return t
    else:
        raise ValueError('Non-trivial rotation not yet implemented')


def t_matrix_collection(vacuum_wavelength, particle_collection, layer_system, index_specs):
    """Return the T-matrices for all particles as a numpy.ndarray, in the format (NS, blocksize, blocksize) where NS is
    the number of particles and blocksize is the number of SWE terms per particle.

    Input:
    vacuum_wavelength:      (length unit)
    particle_collection:    An instance of  smuthi.particles.ParticleCollection describing the scattering particles
    layer_system:           An instance of smuthi.layers.LayerSystem describing the stratified medium
    index_specs:            A dictionary with the entries 'lmax', 'mmax' and 'index arrangement'
    """
    if index_specs['index arrangement'] == 'stlm':
        blocksize = smuthi.index_conversion.block_size(index_specs=index_specs)
        particle_number = particle_collection.particle_number()
        t_matrices = np.zeros((particle_number, blocksize, blocksize), dtype=complex)

        particle_params_table = []
        tmatrix_table = []

        for ip, particle in enumerate(particle_collection.particles):
            # gather relevant parameters and omit those that don't alter the T-matrix (before rotation)
            # in the first place, just omit position and euler angel, but add refractive index of surrounding medium
            if particle['shape'] == 'sphere':
                params = ['sphere', particle['radius'], particle['refractive index']]
            else:
                raise ValueError('invalid particle type: so far only spheres are implemented')
            zS = particle['position'][2]
            iS = layer_system.layer_number(zS)
            n_medium = layer_system.refractive_indices[iS]
            params.append(n_medium)

            # compare parameters to table
            for i_old, old_params in enumerate(particle_params_table):
                if params == old_params:
                    tmatrix = tmatrix_table[i_old]
                    break
            else:  # compute T-matrix and update tables
                tmatrix = t_matrix(vacuum_wavelength, n_medium, particle, index_specs)
                particle_params_table.append(params)
                tmatrix_table.append(tmatrix)

            tmatrix_rotated = rotate_t_matrix(t=tmatrix, euler_angles=particle['euler angles'], index_specs=index_specs)
            t_matrices[ip, :, :] = tmatrix_rotated

    else:
        raise ValueError('invalid index arrangement: currently only "stml" is implemented')

    return t_matrices