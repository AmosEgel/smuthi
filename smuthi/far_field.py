import numpy as np
import smuthi.coordinates as coord
import smuthi.layers as lay
import smuthi.plane_wave_pattern as pwp


def scattering_cross_section(polar_angles=None, initial_field_collection=None, azimuthal_angles=None,
                             particle_collection=None, linear_system=None, layer_system=None,
                             layerresponse_precision=None):
    """Evaluate and display the differential scattering cross section as a function of solid angle.
    Return a dictionary scattering_cross_section with the following entries:
    'differential':     Differential cross section as an np.array of dimension 2 x nb x na where nb is the number of
                        polar angles and na is the number of azimuthal angles. The first index refers to polarization.
    'total':            Total cross section as list with two entries (for two polarizations)
    'polar':            Polar differential cross section (that is scattered power per polar angle, divided by incoming
                        intensity)
    'polar angles':     Polar angles for which the differential cross section was evaluated
    'azimuthal angles': Azimuthal angles for which the differential cross section was evaluated
    'forward indices':  The indices of polar angles that correspond to directions in the top hemisphere
    'backward indices': The indices of polar angles that correspond to directions in the bottom hemisphere

    input:
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

    initial_intensity = abs(A_P) ** 2 * abs(np.cos(beta_P)) * n_P / 2

    # differential scattering cross section
    dscs = scattered_far_field(polar_angles, vacuum_wavelength, azimuthal_angles, particle_collection, linear_system,
                               layer_system, layerresponse_precision)['intensity'] / initial_intensity

    top_idcs = polar_angles <= (np.pi / 2)
    bottom_idcs = polar_angles > (np.pi / 2)

    if not k_top.imag == 0:
        dscs[:, top_idcs, :] = 0
    if not k_bot.imag == 0:
        dscs[:, bottom_idcs, :] = 0

    if len(azimuthal_angles) > 2:
        # azimuthal average
        polar_dscs = np.trapz(dscs, azimuthal_angles[None, None, :]) * np.sin(polar_angles[None, :])

        # total scattering cross section
        total_cs_top = np.trapz(polar_dscs[:, top_idcs], polar_angles[None, top_idcs])
        total_cs_bottom = np.trapz(polar_dscs[:, bottom_idcs], polar_angles[None, bottom_idcs])
        total_cs = total_cs_top + total_cs_bottom
    else:
        polar_dscs = None
        total_cs_top = None
        total_cs_bottom = None
        total_cs = None

    scs = {'differential': dscs, 'total': total_cs, 'total top': total_cs_top,
           'total bottom': total_cs_bottom, 'polar': polar_dscs, 'polar angles': polar_angles,
           'azimuthal angles': azimuthal_angles}

    return scs


def extinction_cross_section(initial_field_collection=None, particle_collection=None, linear_system=None,
                             layer_system=None, layerresponse_precision=None):
    """Evaluate and display the differential scattering cross section as a function of solid angle. Return dictionary
    with entries
    'forward':      Extinction in the positinve z-direction (top layer)
    'backward':     Extinction in the negative z-direction (bottom layer)

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

    gr_scat_bottom_list = pwp.plane_wave_pattern_rs(
        n_effective=np.array([kappa_P/omega]), azimuthal_angles=np.array([alpha_P]),
        vacuum_wavelength=vacuum_wavelength, particle_collection=particle_collection, linear_system=linear_system,
        layer_system=layer_system, layer_numbers=[0])
    g_scat_bottom_list = pwp.plane_wave_pattern_s(
        n_effective=np.array([kappa_P/omega]), azimuthal_angles=np.array([alpha_P]),
        vacuum_wavelength=vacuum_wavelength, particle_collection=particle_collection, linear_system=linear_system,
        layer_system=layer_system, layer_numbers=[0])
    kz_bot = coord.k_z(k_parallel=kappa_P, k=k_bot)
    g_scat_bottom = gr_scat_bottom_list[0][pol_P, 1, 0, 0] + g_scat_bottom_list[0][pol_P, 1, 0, 0]
    P_bot_ext = 4 * np.pi ** 2 * kz_bot / omega * (gRPbot * np.conj(g_scat_bottom)).real
    bottom_extinction_cs = - P_bot_ext / initial_intensity

    # bottom extinction
    Ltop = lay.layersystem_response_matrix(pol_P, layer_system.thicknesses, layer_system.refractive_indices, kappa_P,
                                           omega, i_P, i_top, layerresponse_precision)
    gRPtop = (Ltop[0, 0] + Ltop[0, 1]) * A_P

    gr_scat_top_list = pwp.plane_wave_pattern_rs(
        n_effective=np.array([kappa_P/omega]), azimuthal_angles=np.array([alpha_P]),
        vacuum_wavelength=vacuum_wavelength, particle_collection=particle_collection, linear_system=linear_system,
        layer_system=layer_system, layer_numbers=[i_top])
    g_scat_top_list = pwp.plane_wave_pattern_s(
        n_effective=np.array([kappa_P/omega]), azimuthal_angles=np.array([alpha_P]),
        vacuum_wavelength=vacuum_wavelength, particle_collection=particle_collection, linear_system=linear_system,
        layer_system=layer_system, layer_numbers=[i_top])
    kz_top = coord.k_z(k_parallel=kappa_P, k=k_top)
    g_scat_top = gr_scat_top_list[0][pol_P, 0, 0, 0] + g_scat_top_list[0][pol_P, 0, 0, 0]
    P_top_ext = 4 * np.pi ** 2 * kz_top / omega * (gRPtop * np.conj(g_scat_top)).real
    top_extinction_cs = - P_top_ext / initial_intensity

    extinction_cs = {'top': top_extinction_cs, 'bottom': bottom_extinction_cs}

    return extinction_cs


def scattered_far_field(polar_angles=None, vacuum_wavelength=None, azimuthal_angles=None,
                        particle_collection=None, linear_system=None, layer_system=None, layerresponse_precision=None):
    """Return a far_field dictionary with the following entries:
    'intensity':        Radiant far field intensity as ndarray of shape 2 x nb x na where nb =len(polar_angles) and
                        na = len(azimuthal_angles).
                        The indices are:
                        - polarization (0=TE, 1=TM)
                        - polar angle index
                        - azimuthal angle index
    'polar intensity':  Polar far field, that is the power per polar angle as ndarray of shape 2 x nb
    'top power':        Total scattered power into top layer (positive z direction)
    'bottom power':     Total scattered power into bottom layer (negative z direction)
    'polar angles':     Polar angles
    'azimuthal angles': Azimuthal angles

    input:
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
    top_pwp_rs = pwp.plane_wave_pattern_rs(
        n_effective=neff_top, azimuthal_angles=azimuthal_angles, vacuum_wavelength=vacuum_wavelength,
        particle_collection=particle_collection, linear_system=linear_system, layer_system=layer_system,
        layer_numbers=[i_top], layerresponse_precision=layerresponse_precision)

    top_pwp_s = pwp.plane_wave_pattern_s(
        n_effective=neff_top, azimuthal_angles=azimuthal_angles, vacuum_wavelength=vacuum_wavelength,
        particle_collection=particle_collection, linear_system=linear_system, layer_system=layer_system,
        layer_numbers=[i_top])

    bottom_pwp_rs = pwp.plane_wave_pattern_rs(
        n_effective=neff_bottom, azimuthal_angles=azimuthal_angles, vacuum_wavelength=vacuum_wavelength,
        particle_collection=particle_collection, linear_system=linear_system, layer_system=layer_system,
        layer_numbers=[0], layerresponse_precision=layerresponse_precision)

    bottom_pwp_s = pwp.plane_wave_pattern_s(
        n_effective=neff_bottom, azimuthal_angles=azimuthal_angles, vacuum_wavelength=vacuum_wavelength,
        particle_collection=particle_collection, linear_system=linear_system, layer_system=layer_system,
        layer_numbers=[0])

    pwp_total = np.concatenate([top_pwp_rs[0][:, 0, :, :] + top_pwp_s[0][:, 0, :, :],
                          bottom_pwp_rs[0][:, 1, :, :] + bottom_pwp_s[0][:, 1, :, :]], axis=1)

    k_top = omega * layer_system.refractive_indices[i_top]
    k_0 = omega * layer_system.refractive_indices[0]
    kkz2_top = coord.k_z(n_effective=neff_top, omega=omega, k=k_top) ** 2 * k_top
    kkz2_bottom = coord.k_z(n_effective=neff_bottom, omega=omega, k=k_0) ** 2 * k_0
    kkz2 = np.concatenate([kkz2_top, kkz2_bottom])

    far_field_intensity = (2 * np.pi ** 2 / omega * kkz2[np.newaxis, :, np.newaxis] * abs(pwp_total) ** 2).real

    # azimuthal average
    if len(azimuthal_angles) > 2:
        polar_far_field = np.trapz(far_field_intensity, azimuthal_angles[None, None, :]) * np.sin(polar_angles[None, :])
        # total scattered power
        total_top_power = np.trapz(polar_far_field[:, top_idcs], polar_angles[None, top_idcs])
        total_bottom_power = np.trapz(polar_far_field[:, bottom_idcs], polar_angles[None, bottom_idcs])
    else:
        polar_far_field = None
        total_top_power = None
        total_bottom_power = None

    far_field = {'intensity': far_field_intensity,
                 'polar intensity': polar_far_field,
                 'top power': total_top_power,
                 'bottom power': total_bottom_power,
                 'polar angles': polar_angles,
                 'azimuthal angles': azimuthal_angles}

    return far_field
