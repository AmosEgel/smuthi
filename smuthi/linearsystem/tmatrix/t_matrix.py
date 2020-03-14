# -*- coding: utf-8 -*-
import numpy as np
import smuthi.fields as flds
import smuthi.utility.math as sf
import smuthi.linearsystem.tmatrix.nfmds.t_matrix_axsym as nftaxs
import smuthi.fields.transformations as trf


def mie_coefficient(tau, l, k_medium, k_particle, radius):
    """Return the Mie coefficients of a sphere.

    Args:
        tau         integer: spherical polarization, 0 for spherical TE and 1 for spherical TM
        l           integer: l=1,... multipole degree (polar quantum number)
        k_medium    float or complex: wavenumber in surrounding medium (inverse length unit)
        k_particle  float or complex: wavenumber inside sphere (inverse length unit)
        radius      float: radius of sphere (length unit)

    Returns:
        Mie coefficients as complex
    """
    jlkr_medium = sf.spherical_bessel(l, k_medium * radius)
    jlkr_particle = sf.spherical_bessel(l, k_particle * radius)
    dxxj_medium = sf.dx_xj(l, k_medium * radius)
    dxxj_particle = sf.dx_xj(l, k_particle * radius)

    hlkr_medium = sf.spherical_hankel(l, k_medium * radius)
    dxxh_medium = sf.dx_xh(l, k_medium * radius)

    if tau == 0:
        q = (jlkr_medium * dxxj_particle - jlkr_particle * dxxj_medium) / (jlkr_particle * dxxh_medium - hlkr_medium *
                                                                           dxxj_particle)
    elif tau == 1:
        q = ((k_medium ** 2 * jlkr_medium * dxxj_particle - k_particle ** 2 * jlkr_particle * dxxj_medium) /
             (k_particle ** 2 * jlkr_particle * dxxh_medium - k_medium ** 2 * hlkr_medium * dxxj_particle))
    else:
        raise ValueError('tau must be 0 (spherical TE) or 1 (spherical TM)')

    return q


def internal_mie_coefficient(tau, l, k_medium, k_particle, radius):
    """Return the Mie coefficients to compute the internal field of a sphere.

    Args:
        tau         integer: spherical polarization, 0 for spherical TE and 1 for spherical TM
        l           integer: l=1,... multipole degree (polar quantum number)
        k_medium    float or complex: wavenumber in surrounding medium (inverse length unit)
        k_particle  float or complex: wavenumber inside sphere (inverse length unit)
        radius      float: radius of sphere (length unit)

    Returns:
        Internal Mie coefficients as complex
    """
    jlkr_medium = sf.spherical_bessel(l, k_medium * radius)
    jlkr_particle = sf.spherical_bessel(l, k_particle * radius)
    dxxj_medium = sf.dx_xj(l, k_medium * radius)
    dxxj_particle = sf.dx_xj(l, k_particle * radius)

    hlkr_medium = sf.spherical_hankel(l, k_medium * radius)
    dxxh_medium = sf.dx_xh(l, k_medium * radius)

    if tau == 0:
        q = (jlkr_medium * dxxh_medium - hlkr_medium * dxxj_medium) / (jlkr_particle * dxxh_medium
                                                                       - hlkr_medium * dxxj_particle)
    elif tau == 1:
        q = ((k_medium * k_particle * jlkr_medium * dxxh_medium - k_medium * k_particle * hlkr_medium * dxxj_medium) /
             (k_particle ** 2 * jlkr_particle * dxxh_medium - k_medium ** 2 * hlkr_medium * dxxj_particle))
    else:
        raise ValueError('tau must be 0 (spherical TE) or 1 (spherical TM)')

    return q


def t_matrix_sphere(k_medium, k_particle, radius, l_max, m_max):
    """T-matrix of a spherical scattering object.

    Args:
        k_medium (float or complex):            Wavenumber in surrounding medium (inverse length unit)
        k_particle (float or complex):          Wavenumber inside sphere (inverse length unit)
        radius (float):                         Radius of sphere (length unit)
        l_max (int):                            Maximal multipole degree
        m_max (int):                            Maximal multipole order

    Returns:
         T-matrix as ndarray
    """
    t = np.zeros((flds.blocksize(l_max, m_max), flds.blocksize(l_max, m_max)), dtype=complex)
    for tau in range(2):
        for m in range(-m_max, m_max + 1):
            for l in range(max(1, abs(m)), l_max + 1):
                n = flds.multi_to_single_index(tau, l, m, l_max, m_max)
                t[n, n] = mie_coefficient(tau, l, k_medium, k_particle, radius)
    return t


def t_matrix(vacuum_wavelength, n_medium, particle):
    """Return the T-matrix of a particle.

    Args:
        vacuum_wavelength(float)
        n_medium(float or complex):             Refractive index of surrounding medium
        particle(smuthi.particles.Particle):    Particle object

    Returns:
        T-matrix as ndarray
    """
    if type(particle).__name__ == 'Sphere':
        k_medium = 2 * np.pi / vacuum_wavelength * n_medium
        k_particle = 2 * np.pi / vacuum_wavelength * particle.refractive_index
        radius = particle.radius
        t = t_matrix_sphere(k_medium, k_particle, radius, particle.l_max, particle.m_max)
    elif type(particle).__name__ == 'Spheroid':
        t = nftaxs.tmatrix_spheroid(vacuum_wavelength=vacuum_wavelength, layer_refractive_index=n_medium,
                                    particle_refractive_index=particle.refractive_index,
                                    semi_axis_c=particle.semi_axis_c, semi_axis_a=particle.semi_axis_a,
                                    use_ds=particle.t_matrix_method.get('use discrete sources', True),
                                    nint=particle.t_matrix_method.get('nint', 200),
                                    nrank=particle.t_matrix_method.get('nrank', particle.l_max + 2),
                                    l_max=particle.l_max, m_max=particle.m_max)
        if not particle.euler_angles == [0, 0, 0]:
            t = rotate_t_matrix(t, particle.l_max, particle.m_max, particle.euler_angles, wdsympy=False)
    elif type(particle).__name__ == 'FiniteCylinder':
        t = nftaxs.tmatrix_cylinder(vacuum_wavelength=vacuum_wavelength, layer_refractive_index=n_medium,
                                    particle_refractive_index=particle.refractive_index,
                                    cylinder_height=particle.cylinder_height,
                                    cylinder_radius=particle.cylinder_radius,
                                    use_ds=particle.t_matrix_method.get('use discrete sources', True),
                                    nint=particle.t_matrix_method.get('nint', 200),
                                    nrank=particle.t_matrix_method.get('nrank', particle.l_max + 2),
                                    l_max=particle.l_max, m_max=particle.m_max)
        if not particle.euler_angles == [0, 0, 0]:
            t = rotate_t_matrix(t, particle.l_max, particle.m_max, particle.euler_angles, wdsympy=False)
    else:
        raise ValueError('T-matrix for ' + type(particle).__name__ + ' currently not implemented.')

    return t


def rotate_t_matrix(T, l_max, m_max, euler_angles, wdsympy=False):
    """T-matrix of a rotated particle.

    Args:
        T (numpy.array):        T-matrix
        l_max (int):            Maximal multipole degree
        m_max (int):            Maximal multipole order
        euler_angles (list):    Euler angles [alpha, beta, gamma] of rotated particle in (zy'z''-convention) in radian

    Returns:
        rotated T-matrix (numpy.array)
    """

    if euler_angles == [0, 0, 0]:
        return T
    else:
        # Doicu, Light Scattering by Systems of Particles, p. 70 (1.115)
        rot_mat_1 = trf.block_rotation_matrix_D_svwf(l_max, m_max, -euler_angles[2], -euler_angles[1],
                                                     -euler_angles[0], wdsympy)
        rot_mat_2 = trf.block_rotation_matrix_D_svwf(l_max, m_max, euler_angles[0], euler_angles[1], euler_angles[2],
                                                     wdsympy)
        T_mat_rot = (np.dot(np.dot(np.transpose(rot_mat_1), T), np.transpose(rot_mat_2)))

        # Mishchenko, Scattering, Absorption and Emission of Light by small Particles, p.120 (5.29)
        #       T_rot_matrix = np.dot(np.dot(trf.rotation_matrix_D(l_max, alpha, beta, gamma), T),
        #                             trf.rotation_matrix_D(l_max, -gamma, -beta, -alpha))

        return T_mat_rot
