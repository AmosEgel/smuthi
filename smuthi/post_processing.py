# -*- coding: utf-8 -*-
import numpy as np
import smuthi.coordinates as coord
import smuthi.index_conversion as idx
import smuthi.vector_wave_functions as vwf
import smuthi.layers as lay
import matplotlib.pyplot as plt


class PostProcessing:
    def __init__(self):
        self.tasks = []

    def run(self, simulation):
        particle_collection = simulation.particle_collection
        linear_system = simulation.linear_system
        layer_system = simulation.layer_system
        for item in self.tasks:
            if item['task'] == 'evaluate cross sections':
                polar_angles = item.get('polar angles')
                azimuthal_angles = item.get('azimuthal angles')
                layerresponse_precision = item.get('layerresponse precision')
                filename_forward = item.get('filename forward')
                filename_backward = item.get('filename backward')
                differential_scattering_cross_section(polar_angles=polar_angles,
                                                      initial_field_collection=simulation.initial_field_collection,
                                                      azimuthal_angles=azimuthal_angles,
                                                      particle_collection=particle_collection,
                                                      linear_system=linear_system, layer_system=layer_system,
                                                      layerresponse_precision=layerresponse_precision,
                                                      filename_forward=filename_forward,
                                                      filename_backward=filename_backward,
                                                      length_unit=simulation.length_unit)
                extinction_cross_section(initial_field_collection=simulation.initial_field_collection,
                                         particle_collection=particle_collection, linear_system=linear_system,
                                         layer_system=layer_system, layerresponse_precision=layerresponse_precision,
                                         length_unit=simulation.length_unit)


def differential_scattering_cross_section(polar_angles=None, initial_field_collection=None, azimuthal_angles=None,
                                          particle_collection=None, linear_system=None, layer_system=None,
                                          layerresponse_precision=None, filename_forward=None, filename_backward=None,
                                          length_unit=None):
    """Evaluate and display the differential scattering cross section as a function of solid angle.

    polar_angles:               (float) array of polar angles (radian), default: from 1 to 180 degree in steps of 1
    initial_field_collection:   smuthi.initial_field.InitialFieldCollection object
    azimuthal_angles:           (float) array of azimuthal angles (radian), default: from 1 to 360 degree in steps of 1
    particle_collection:        smuthi.particles.ParticleCollection object
    linear_system:              smuthi.linear_system.LinearSystem object
    layer_system:               smuthi.layers.LayerSystem object
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    if (len(initial_field_collection.specs_list) > 1
         or not initial_field_collection.specs_list[0]['type'] == 'plane wave'):
        raise ValueError('Cross section only defined for single plane wave excitation.')

    if polar_angles is None:
        polar_angles = (np.concatenate([np.arange(0, 90, 1, dtype=float), np.arange(91, 181, 1, dtype=float)])
                        * np.pi / 180)
    if azimuthal_angles is None:
        azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180

    i_top = layer_system.number_of_layers() - 1
    vacuum_wavelength = initial_field_collection.vacuum_wavelength
    omega = coord.angular_frequency(vacuum_wavelength)
    k_bot = omega * layer_system.refractive_indices[0]
    k_top = omega * layer_system.refractive_indices[-1]

    # read plane wave parameters
    A_P = initial_field_collection.specs_list[0]['amplitude']
    beta_P = initial_field_collection.specs_list[0]['polar angle']
    if beta_P < np.pi / 2:
        i_P = 0
        n_P = layer_system.refractive_indices[i_P]
    else:
        i_P = i_top
        n_P = layer_system.refractive_indices[i_P]
    if n_P.imag:
        raise ValueError('plane wave from absorbing layer: cross section undefined')
    else:
        n_P = n_P.real

    kappa_P = np.sin(beta_P) * n_P * omega
    initial_intensity = abs(A_P) ** 2 * abs(np.cos(beta_P)) * n_P / 2

    # differential scattering cross section
    dsc = scattered_far_field(polar_angles, vacuum_wavelength, azimuthal_angles, particle_collection, linear_system,
                              layer_system, layerresponse_precision) / initial_intensity

    top_idcs = polar_angles <= (np.pi / 2)
    bottom_idcs = polar_angles > (np.pi / 2)

    if not k_top.imag == 0:
        dsc[:, top_idcs, :] = 0
    if not k_bot.imag == 0:
        dsc[:, bottom_idcs, :] = 0

    # azimuthal average
    polar_dsc = np.trapz(dsc, azimuthal_angles[None, None, :]) * np.sin(polar_angles[None, :])

    # total scattering cross section
    total_cs = np.trapz(polar_dsc, polar_angles[None, :])
    print(total_cs[0] + total_cs[1])

    """
    # plot -------------------------------------------------------------------------------------------------------------
    # forward
    alpha_grid, beta_grid = np.meshgrid(azimuthal_angles, polar_angles[top_idcs].real * 180 / np.pi)
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.pcolormesh(alpha_grid, beta_grid, dsc[0, top_idcs, :] + dsc[1, top_idcs, :])
    plt.title('forward far field (differential cross section in ' + length_unit + '^2)')
    if filename_forward is not None:
        fig.savefig(filename_forward)

    # backward
    alpha_grid, beta_grid = np.meshgrid(azimuthal_angles, polar_angles[bottom_idcs].real * 180 / np.pi)
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.pcolormesh(alpha_grid, 180 - beta_grid, dsc[0, bottom_idcs, :] + dsc[1, bottom_idcs, :])
    plt.title('backward far field (differential cross section in ' + length_unit + '^2)')
    if filename_backward is not None:
        fig.savefig(filename_backward)

    # averaged
    plt.figure()
    plt.plot(polar_angles * 180 / np.pi, polar_dsc[0, :] + polar_dsc[1, :])
    plt.plot(polar_angles * 180 / np.pi, polar_dsc[0, :])
    plt.plot(polar_angles * 180 / np.pi, polar_dsc[1, :])
    plt.title('polar differential cross section')
    plt.xlabel('polar angle (deg)')
    plt.ylabel('polar differential cross section (' + length_unit + '^2)')
    #-------------------------------------------------------------------------------------------------------------------
    """
    return azimuthal_angles, polar_angles, dsc


def extinction_cross_section(initial_field_collection=None, particle_collection=None, linear_system=None,
                             layer_system=None, layerresponse_precision=None, length_unit=None):
    """Evaluate and display the differential scattering cross section as a function of solid angle.

    polar_angles:               (float) array of polar angles (radian), default: from 1 to 180 degree in steps of 1
    initial_field_collection:   smuthi.initial_field.InitialFieldCollection object
    azimuthal_angles:           (float) array of azimuthal angles (radian), default: from 1 to 360 degree in steps of 1
    particle_collection:        smuthi.particles.ParticleCollection object
    linear_system:              smuthi.linear_system.LinearSystem object
    layer_system:               smuthi.layers.LayerSystem object
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    if (len(initial_field_collection.specs_list) > 1
         or not initial_field_collection.specs_list[0]['type'] == 'plane wave'):
        raise ValueError('Cross section only defined for single plane wave excitation.')

    i_top = layer_system.number_of_layers() - 1
    vacuum_wavelength = initial_field_collection.vacuum_wavelength
    omega = coord.angular_frequency(vacuum_wavelength)
    k_bot = omega * layer_system.refractive_indices[0]
    k_top = omega * layer_system.refractive_indices[-1]

    # read plane wave parameters
    pol_P = initial_field_collection.specs_list[0]['polarization']
    beta_P = initial_field_collection.specs_list[0]['polar angle']
    alpha_P = initial_field_collection.specs_list[0]['azimuthal angle']

    if beta_P < np.pi / 2:
        i_P = 0
        n_P = layer_system.refractive_indices[i_P]
        k_P = k_bot
    else:
        i_P = i_top
        n_P = layer_system.refractive_indices[i_P]
        k_P = k_top
    if n_P.imag:
        raise ValueError('plane wave from absorbing layer: cross section undefined')
    else:
        n_P = n_P.real

    # complex amplitude of initial wave (including phase factor for reference point)
    kappa_P = np.sin(beta_P) * k_P
    kx = np.cos(alpha_P) * kappa_P
    ky = np.sin(alpha_P) * kappa_P
    pm_kz_P = k_P * np.cos(beta_P)
    kvec_P = np.array([kx, ky, pm_kz_P])
    rvec_iP = np.array([0, 0, layer_system.reference_z(i_P)])
    rvec_0 = np.array(initial_field_collection.specs_list[0]['reference point'])
    ejkriP = np.exp(1j * np.dot(kvec_P, rvec_iP - rvec_0))
    A_P = initial_field_collection.specs_list[0]['amplitude'] * ejkriP

    initial_intensity = abs(A_P) ** 2 * abs(np.cos(beta_P)) * n_P / 2

    # bottom extinction
    Lbot = lay.layersystem_response_matrix(pol_P, layer_system.thicknesses, layer_system.refractive_indices, kappa_P,
                                           omega, i_P, 0, layerresponse_precision)
    gRPbot = (Lbot[1, 0] + Lbot[1, 1]) * A_P

    gr_scat_bottom_list = plane_wave_pattern_rs(n_effective=np.array([kappa_P/omega]),
                                                azimuthal_angles=np.array([alpha_P]),
                                                vacuum_wavelength=vacuum_wavelength,
                                                particle_collection=particle_collection,
                                                linear_system=linear_system, layer_system=layer_system,
                                                layer_numbers=[0])
    g_scat_bottom_list = plane_wave_pattern_s(n_effective=np.array([kappa_P/omega]),
                                              azimuthal_angles=np.array([alpha_P]), vacuum_wavelength=vacuum_wavelength,
                                              particle_collection=particle_collection, linear_system=linear_system,
                                              layer_system=layer_system, layer_numbers=[0])
    kz_bot = coord.k_z(k_parallel=kappa_P, k=k_bot)
    g_scat_bottom = gr_scat_bottom_list[0][pol_P, 1, 0, 0] + g_scat_bottom_list[0][pol_P, 1, 0, 0]
    P_bot_ext = -4 * np.pi ** 2 * kz_bot / omega * (gRPbot * np.conj(g_scat_bottom)).real
    bottom_extinction_cs = P_bot_ext / initial_intensity
    print(bottom_extinction_cs)

    # bottom extinction
    Ltop = lay.layersystem_response_matrix(pol_P, layer_system.thicknesses, layer_system.refractive_indices, kappa_P,
                                           omega, i_P, i_top, layerresponse_precision)
    gRPtop = (Ltop[0, 0] + Ltop[0, 1]) * A_P

    gr_scat_top_list = plane_wave_pattern_rs(n_effective=np.array([kappa_P/omega]),
                                             azimuthal_angles=np.array([alpha_P]), vacuum_wavelength=vacuum_wavelength,
                                             particle_collection=particle_collection, linear_system=linear_system,
                                             layer_system=layer_system, layer_numbers=[i_top])
    g_scat_top_list = plane_wave_pattern_s(n_effective=np.array([kappa_P/omega]), azimuthal_angles=np.array([alpha_P]),
                                           vacuum_wavelength=vacuum_wavelength, particle_collection=particle_collection,
                                           linear_system=linear_system, layer_system=layer_system,
                                           layer_numbers=[i_top])
    kz_top = coord.k_z(k_parallel=kappa_P, k=k_top)
    g_scat_top = gr_scat_top_list[0][pol_P, 0, 0, 0] + g_scat_top_list[0][pol_P, 0, 0, 0]
    P_top_ext = 4 * np.pi ** 2 * kz_top / omega * (gRPtop * np.conj(g_scat_top)).real
    top_extinction_cs = P_top_ext / initial_intensity
    print(top_extinction_cs)

    """print(abs(gr_scat_bottom_list[0][pol_P, 1, 0, 0]))
    print(abs(g_scat_bottom_list[0][pol_P, 1, 0, 0]))
    print(abs(g_scat_bottom_list[0][pol_P, 1, 0, 0] + gr_scat_bottom_list[0][pol_P, 1, 0, 0]))

    print(abs(gr_scat_top_list[0][pol_P, 0, 0, 0]))
    print(abs(g_scat_top_list[0][pol_P, 0, 0, 0]))
    print(abs(g_scat_top_list[0][pol_P, 0, 0, 0] + gr_scat_top_list[0][pol_P, 0, 0, 0]))"""

    #print(g_scat_top_list[0][pol_P, 0, 0, 0] / gRPtop)
    #print(gr_scat_top_list[0][pol_P, 0, 0, 0]/ gRPtop)



def show_scattered_far_field(polar_angles=None, initial_field_collection=None, azimuthal_angles=None,
                             particle_collection=None, linear_system=None, layer_system=None,
                             layerresponse_precision=None, filename_forward=None, filename_backward=None,
                             length_unit=None):
    """Plot the far field intensity. For plane waves as initial field, it is displayed as a differential cross section.

    polar_angles:               (float) array of polar angles (radian), default: from 1 to 180 degree in steps of 1
    initial_field_collection:   smuthi.initial_field.InitialFieldCollection object
    azimuthal_angles:           (float) array of azimuthal angles (radian), default: from 1 to 360 degree in steps of 1
    particle_collection:        smuthi.particles.ParticleCollection object
    linear_system:              smuthi.linear_system.LinearSystem object
    layer_system:               smuthi.layers.LayerSystem object
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    if polar_angles is None:
        polar_angles = np.arange(0, 181, 1, dtype=float) * np.pi / 180
    if azimuthal_angles is None:
        azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180

    vacuum_wavelength = initial_field_collection.vacuum_wavelength
    far_field = scattered_far_field(polar_angles, vacuum_wavelength, azimuthal_angles, particle_collection,
                                    linear_system, layer_system, layerresponse_precision)

    top_idcs = polar_angles <= (np.pi / 2)
    bottom_idcs = polar_angles > (np.pi / 2)

    if len(initial_field_collection.specs_list) == 1:
        initial_field_specs = initial_field_collection.specs_list[0]
        if initial_field_specs['type'] == 'plane wave':  # in that case, differential cross section
            if initial_field_specs['polar angle'] < np.pi / 2:
                n_inc = layer_system.refractive_indices[0]
            else:
                n_inc = layer_system.refractive_indices[-1]

        # forward
        alpha_grid, beta_grid = np.meshgrid(azimuthal_angles, polar_angles[top_idcs].real * 180 / np.pi)

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.pcolormesh(alpha_grid, beta_grid, (far_field[0, top_idcs, :] + far_field[1, top_idcs, :]) * n_inc / 2)
        plt.title('forward far field (differential cross section in ' + length_unit + '^2)')
        if filename_forward is not None:
            fig.savefig(filename_forward)

        # backward
        alpha_grid, beta_grid = np.meshgrid(azimuthal_angles, polar_angles[bottom_idcs].real * 180 / np.pi)

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.pcolormesh(alpha_grid, 180 - beta_grid,
                      (far_field[0, bottom_idcs, :] + far_field[1, bottom_idcs, :]) * n_inc / 2)
        plt.title('backward far field (differential cross section in ' + length_unit + '^2)')
        if filename_backward is not None:
            fig.savefig(filename_backward)


def scattered_far_field(polar_angles=None, vacuum_wavelength=None, azimuthal_angles=None,
                        particle_collection=None, linear_system=None, layer_system=None, layerresponse_precision=None):
    """Radiant far field intensity as ndarray of shape 2 x nb x na where nb =len(polar_angles) and
    na = len(azimuthal_angles).
    The indices are:
    - polarization (0=TE, 1=TM)
    - polar angle index
    - azimuthal angle index

    TODO: TEST

    polar_angles:           (float) array of polar angles values (radian)
    vacuum_wavelength:      (float, length unit)
    azimuthal_angles:       (float) array of azimuthal angle values (radian)
    particle_collection:    smuthi.particles.ParticleCollection object
    linear_system:          smuthi.linear_system.LinearSystem object
    layer_system:           smuthi.layers.LayerSystem object
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    omega = coord.angular_frequency(vacuum_wavelength)
    if polar_angles is None:
        polar_angles = np.arange(0, 181, 1, dtype=float) * np.pi / 180
    if azimuthal_angles is None:
        azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180

    i_top = layer_system.number_of_layers() - 1
    top_idcs = polar_angles <= (np.pi / 2)
    bottom_idcs = polar_angles > (np.pi / 2)
    neff_top = np.sin(polar_angles[top_idcs]) * layer_system.refractive_indices[i_top]
    neff_bottom = np.sin(polar_angles[bottom_idcs]) * layer_system.refractive_indices[0]

    top_pwp_rs = plane_wave_pattern_rs(n_effective=neff_top, azimuthal_angles=azimuthal_angles,
                                       vacuum_wavelength=vacuum_wavelength, particle_collection=particle_collection,
                                       linear_system=linear_system, layer_system=layer_system, layer_numbers=[i_top],
                                       layerresponse_precision=layerresponse_precision)

    top_pwp_s = plane_wave_pattern_s(n_effective=neff_top, azimuthal_angles=azimuthal_angles,
                                     vacuum_wavelength=vacuum_wavelength, particle_collection=particle_collection,
                                     linear_system=linear_system, layer_system=layer_system, layer_numbers=[i_top])

    bottom_pwp_rs = plane_wave_pattern_rs(n_effective=neff_bottom, azimuthal_angles=azimuthal_angles,
                                          vacuum_wavelength=vacuum_wavelength, particle_collection=particle_collection,
                                          linear_system=linear_system, layer_system=layer_system, layer_numbers=[0],
                                          layerresponse_precision=layerresponse_precision)

    bottom_pwp_s = plane_wave_pattern_s(n_effective=neff_bottom, azimuthal_angles=azimuthal_angles,
                                        vacuum_wavelength=vacuum_wavelength, particle_collection=particle_collection,
                                        linear_system=linear_system, layer_system=layer_system, layer_numbers=[0])

    pwp = np.concatenate([top_pwp_rs[0][:, 0, :, :] + top_pwp_s[0][:, 0, :, :],
                          bottom_pwp_rs[0][:, 1, :, :] + bottom_pwp_s[0][:, 1, :, :]], axis=1)

    k_top = omega * layer_system.refractive_indices[i_top]
    k_0 = omega * layer_system.refractive_indices[0]
    kkz2_top = coord.k_z(n_effective=neff_top, omega=omega, k=k_top) ** 2 * k_top
    kkz2_bottom = coord.k_z(n_effective=neff_bottom, omega=omega, k=k_0) ** 2 * k_0
    kkz2 = np.concatenate([kkz2_top, kkz2_bottom])

    far_field = (2 * np.pi ** 2 / omega * kkz2[np.newaxis, :, np.newaxis] * abs(pwp) ** 2).real
    return far_field


def plane_wave_pattern_rs(n_effective=None, azimuthal_angles=None, vacuum_wavelength=None,
                          particle_collection=None, linear_system=None, layer_system=None, layer_numbers=None,
                          layerresponse_precision=None):
    """Layer system response plane wave pattern of the scattered field.
    Return a list of plane wave patterns as ndarrays of shape 2 x 2 x nk x na where nk =len(n_effective) and
    na = len(azimuthal_angles).
    The indices are:
    - polarization (0=TE, 1=TM)
    - up/down (0=upwards propagating +kz, 1=downwards propagation -kz)
    - n_effective index
    - azimuthal angle index
    A plane wave pattern is the integrand of a 2-dimensional expansion of the layer system
    response to the scattered field in PVWFs.

    TODO: TEST

    n_effective:            (float) array of n_eff values, i.e. k_parallel / omega
    azimuthal_angles:       (float) array of azimuthal angle values (radian)
    vacuum_wavelength:      (float, length unit)
    particle_collection:    smuthi.particles.ParticleCollection object
    linear_system:          smuthi.linear_system.LinearSystem object
    layer_system:           smuthi.layers.LayerSystem object
    layer_numbers:          list of layer numbers in which the plane wave pattern shall be computed
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    omega = coord.angular_frequency(vacuum_wavelength)
    neff_grid, azimuthal_angle_grid = np.meshgrid(n_effective, azimuthal_angles, indexing='ij')
    kpar_grid = neff_grid * omega

    # read out index specs
    lmax = idx.l_max
    mmax = idx.m_max
    blocksize = idx.number_of_indices()

    iS_list = [layer_system.layer_number(prtcl['position'][2]) for prtcl in particle_collection.particles]
    iS_unique = list(set(iS_list))

    kpar = n_effective * omega
    kiS = [None] * layer_system.number_of_layers()
    kziS = [None] * layer_system.number_of_layers()
    ziS = [None] * layer_system.number_of_layers()
    # transformation coefficients
    B = [None] * layer_system.number_of_layers()
    # layer response
    L = [[None] * layer_system.number_of_layers()] * layer_system.number_of_layers() # indices: from layer, to layer

    m_vec = np.zeros(blocksize, dtype=int)
    for i in iS_unique:
        kiS[i] = omega * layer_system.refractive_indices[i]
        kziS[i] = coord.k_z(k_parallel=kpar, k=kiS[i])
        ziS[i] = layer_system.reference_z(i)
        # transformation coefficients
        B[i] = np.zeros((2, 2, blocksize, len(n_effective)), dtype=complex)  # indices: pol, plus/minus, n, kpar_idx

        for tau in range(2):
            for m in range(-mmax, mmax + 1):
                for l in range(max(1, abs(m)), lmax + 1):
                    n = idx.multi_to_single_index(tau, l, m)
                    m_vec[n] = m
                    for pol in range(2):
                        B[i][pol, 0, n, :] = vwf.transformation_coefficients_VWF(tau, l, m, pol, kpar, kziS[i])
                        B[i][pol, 1, n, :] = vwf.transformation_coefficients_VWF(tau, l, m, pol, kpar, -kziS[i])

        for i_to in layer_numbers:
            L[i][i_to] = np.zeros((2, 2, 2, len(n_effective)), dtype=complex)  # idcs: pol, pl/mn1, pl/mn2, kpar_idx
            for pol in range(2):
                    L[i][i_to][pol, :, :, :] = lay.layersystem_response_matrix(pol, layer_system.thicknesses,
                                                                               layer_system.refractive_indices, kpar,
                                                                               omega, i, i_to, layerresponse_precision)

    pwp = np.zeros((2, 2, len(n_effective), len(azimuthal_angles)), dtype=complex)  # pol, pl/mn, kp_idx, alpha_idx
    pwp_list = [pwp] * len(layer_numbers)
    eima = np.exp(1j * np.tensordot(m_vec, azimuthal_angles, axes=0))  # indices: n, alpha_idx

    for iprt, prtcl in enumerate(particle_collection.particles):
        i = layer_system.layer_number(prtcl['position'][2])
        pos = prtcl['position']
        emnikrs_grid = np.exp(-1j * kpar_grid * (pos[0] * np.cos(azimuthal_angle_grid)  # indices: kpar_idx, alpha_idx
                                                 + pos[1] * np.sin(azimuthal_angle_grid)))
        ziSS = pos[2] - ziS[i]
        emnikz = np.exp(-1j * kziS[i] * ziSS)                       # indices: kpar_idx
        eikz = np.exp(1j * kziS[i] * ziSS)                          # indices: kpar_idx

        fac0 = 1 / (2 * np.pi) * emnikrs_grid                        # indices: kpar_idx, alpha_idx
        fac1 = 1 / (kziS[i] * kiS[i])                                # index: kpar_idx
        b = linear_system.scattered_field_coefficients[np.newaxis, iprt, :, np.newaxis]      # index: 1,n, 1
        beB = B[i]

        beB[:, 0, :, :] = b * (emnikz * beB[:, 0, :, :])
        beB[:, 1, :, :] = b * (eikz * beB[:, 1, :, :])                 # idcs: pol, pl/mn, n, kpar_idx
        eimabeB = np.tensordot(beB, eima, axes=[2, 0])               # idcs: pol, pl/mn, kpar_idx, alpha_idx

        for i_ln, i_to in enumerate(layer_numbers):
            for plmn1 in range(2):
                for plmn2 in range(2):
                    pwp_list[i_ln][:, plmn1, :, :] += (fac0[np.newaxis, :, :] *
                                                       (fac1[np.newaxis, :, np.newaxis] *
                                                        (L[i][i_to][:, plmn1, plmn2, :, np.newaxis] *
                                                         eimabeB[:, plmn2, :, :])))

    return pwp_list


def plane_wave_pattern_s(n_effective=None, azimuthal_angles=None, vacuum_wavelength=None, particle_collection=None,
                         linear_system=None, layer_system=None, layer_numbers=None):
    """Plane wave pattern of the direct scattered field.
    Return a list of plane wave patterns as ndarrays of shape 2 x 2 x nk x na where nk =len(n_effective) and
    na = len(azimuthal_angles).
    The indices are:
    - polarization (0=TE, 1=TM)
    - up/down (0=upwards propagating +kz, 1=downwards propagation -kz)
    - n_effective index
    - azimuthal angle index
    A plane wave pattern is the integrand of a 2-dimensional expansion of the direct
    scattered field in PVWFs. In each layer, the plane wave pattern is only valid below all particles (minus-component)
    or above all particles (plus-component)

    TODO: CHECK IF VALID!!


    n_effective:            (float) array of n_eff values, i.e. k_parallel / omega
    azimuthal_angles:       (float) array of azimuthal angle values (radian)
    vacuum_wavelength:      (float, length unit)
    particle_collection:    smuthi.particles.ParticleCollection object
    linear_system:          smuthi.linear_system.LinearSystem object
    layer_system:           smuthi.layers.LayerSystem object
    layer_numbers:          list of layer numbers in which the plane wave pattern shall be computed
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    omega = coord.angular_frequency(vacuum_wavelength)
    neff_grid, azimuthal_angle_grid = np.meshgrid(n_effective, azimuthal_angles, indexing='ij')
    kpar_grid = neff_grid * omega

    # read out index specs
    lmax = idx.l_max
    mmax = idx.m_max
    blocksize = idx.number_of_indices()

    kpar = n_effective * omega

    # plane wave pattern
    pwp = np.zeros((2, 2, len(n_effective), len(azimuthal_angles)), dtype=complex)  # pol, pl/mn, kp_idx, alpha_idx
    pwp_list = [pwp] * len(layer_numbers)

    m_vec = np.zeros(blocksize, dtype=int)
    for tau in range(2):
        for m in range(-mmax, mmax + 1):
            for l in range(max(1, abs(m)), lmax + 1):
                n = idx.multi_to_single_index(tau, l, m)
                m_vec[n] = m
    eima = np.exp(1j * np.tensordot(m_vec, azimuthal_angles, axes=0))  # indices: n, alpha_idx

    for iidx, i in enumerate(layer_numbers):
        kiS = omega * layer_system.refractive_indices[i]
        kziS = coord.k_z(k_parallel=kpar, k=kiS)
        ziS = layer_system.reference_z(i)
        # transformation coefficients
        B = np.zeros((2, 2, blocksize, len(n_effective)), dtype=complex)  # indices are: pol, plus/minus, n, kpar_idx

        for tau in range(2):
            for m in range(-mmax, mmax + 1):
                for l in range(max(1, abs(m)), lmax + 1):
                    n = idx.multi_to_single_index(tau, l, m)
                    for pol in range(2):
                        B[pol, 0, n, :] = vwf.transformation_coefficients_VWF(tau, l, m, pol, kpar, kziS)
                        B[pol, 1, n, :] = vwf.transformation_coefficients_VWF(tau, l, m, pol, kpar, -kziS)

        for iprt, prtcl in enumerate(particle_collection.particles):
            if layer_system.layer_number(prtcl['position'][2]) == i:
                pos = prtcl['position']
                kx_grid = kpar_grid * np.cos(azimuthal_angle_grid)
                ky_grid = kpar_grid * np.sin(azimuthal_angle_grid)
                emnikplrs_grid = np.exp(-1j * (kx_grid * pos[0] + ky_grid * pos[1]
                                               + kziS[:, np.newaxis] * (pos[2] - ziS)))
                emnikmnrs_grid = np.exp(-1j * (kx_grid * pos[0] + ky_grid * pos[1]
                                               - kziS[:, np.newaxis] * (pos[2] - ziS)))
                eirks = np.concatenate([emnikplrs_grid[np.newaxis, np.newaxis, :, :],
                                        emnikmnrs_grid[np.newaxis, np.newaxis, :, :]], axis=1)  # idcs: kp, al

                fac1 = 1 / (2 * np.pi * kziS * kiS)                          # index: kpar_idx

                b = linear_system.scattered_field_coefficients[iprt, :]     # index: n

                bB = b[np.newaxis, np.newaxis, :, np.newaxis] * B       # idcs: pol, pl/mn, n, kp

                pwp = fac1[np.newaxis, np.newaxis, :, np.newaxis] * eirks * np.tensordot(bB, eima, axes=[2, 0])
                pwp_list[iidx] += pwp

    return pwp_list
