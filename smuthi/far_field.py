import numpy as np
import smuthi.coordinates as coord
import smuthi.field_expansion as fldex


class FarField:
    """Represent the far field intensity of an electromagnetic field.

    If called without specifying the plane_wave_expansion argument, the vacuum_wavelength argument is ignored and
    a far field of NaN's is initialized. Default angular resolution is 1 degree.

    If called with plane_wave_expansion argument, the polar_angles and azimuthal_angles arguments are ignored, and a far
    field according to the plane wave expansion object is calculated.

    Args:
        polar_angles (array):       Polar angles (default: from 0 to 180 degree in steps of 1 degree)
        azimuthal_angles (array):   Azimuthal angles (default: from 0 to 360 degree in steps of 1 degree)
        vacuum_wavelength (float):  Vacuum wavelength
        plane_wave_expansion (smuthi.field_expansion.PlaneWaveExpansion):   Plane wave expansion for which the far field
                                                                            power is computed
    """
    def __init__(self, polar_angles=None, azimuthal_angles=None, vacuum_wavelength=None, plane_wave_expansion=None):
        if plane_wave_expansion is None:
            if polar_angles is None:
                polar_angles = np.arange(0, 181, 1, dtype=float) * np.pi / 180
            if azimuthal_angles is None:
                azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180
            self.polar_angles = polar_angles
            self.azimuthal_angles = azimuthal_angles

            # The far field intensity is represented as a 3-dimensional numpy.ndarray.
            # The indices are:
            # -  polarization (0=TE, 1=TM)
            # - index of the polar angle
            # - index of the azimuthal angle
            self.intensity = np.zeros(2, polar_angles.__len__, azimuthal_angles.__len__)  # dimensions: j, beta, alpha
            self.intensity.fill(np.nan)
        else:
            omega = coord.angular_frequency(vacuum_wavelength)
            k = plane_wave_expansion.k

            if plane_wave_expansion.type == 'upgoing':
                kp = plane_wave_expansion.k_parallel
                self.polar_angles = np.arcsin(kp / k)
            elif plane_wave_expansion.type == 'downgoing':
                kp = plane_wave_expansion.k_parallel[-1::-1]
                self.polar_angles = np.pi - np.arcsin(kp / k)
            else:
                raise ValueError('PWE type not specified')

            if any(polar_angles.imag):
                raise ValueError('complex angles are not allowed')

            self.azimuthal_angles = plane_wave_expansion.azimuthal_angles

            kkz2 = coord.k_z(k_parallel=kp, k=k) ** 2 * k
            self.intensity = (2 * np.pi ** 2 / omega * kkz2[np.newaxis, :, np.newaxis]
                              * abs(plane_wave_expansion.coefficients) ** 2).real

    def azimuthally_averaged_intensity(self):
        if len(self.azimuthal_angles) > 2:
            return np.trapz(self.intensity, self.azimuthal_angles[None, None, :]) * np.sin(self.polar_angles[None, :])
        else:
            return None

    def power(self):
        if len(self.azimuthal_angles) > 2:
            return np.trapz(self.azimuthally_averaged_intensity(), self.polar_angles[None, :])
        else:
            return None


def propagated_initial_far_field(initial_field, layer_system):
    """Evaluate the far field intensity of the reflected / transmitted initial field.

    Args:
        initial_field (smuthi.initial_field.InitialField):  Initial field object of which the far field should be
                                                            evaluated
        layer_system (smuthi.layers.LayerSystem):           Stratified medium

    Returns:
        A tuple of FarField objects, one for forward (i.e., into the top hemisphere) and one for backward propagation
        (bottom hemisphere).
    """
    if not type(initial_field).__name__ == 'GaussianBeam':
        raise ValueError('invalid initial field type')
    i_top = layer_system.number_of_layers() - 1
    top_far_field = FarField(vacuum_wavelength=initial_field.vacuum_wavelength,
                             plane_wave_expansion=initial_field.plane_wave_expansion(layer_system, i_top)[0])
    bottom_far_field = FarField(vacuum_wavelength=initial_field.vacuum_wavelength,
                                plane_wave_expansion=initial_field.plane_wave_expansion(layer_system, 0)[1])

    return top_far_field, bottom_far_field


def initial_intensity(initial_field, layer_system):
    """Evaluate the incoming intensity of the initial field.

    Args:
        initial_field (smuthi.initial_field.GaussianBeam):  Gaussian beam object of which the intensity should be
                                                            evaluated
        layer_system (smuthi.layers.LayerSystem):           Stratified medium

    Returns:
        A FarField object holding the initial intensity information.
    """
    if not type(initial_field).__name__ == 'GaussianBeam':
        raise ValueError('invalid initial field type')

    if np.cos(initial_field.polar_angle) > 0:  # bottom illumination
        far_field = FarField(vacuum_wavelength=initial_field.vacuum_wavelength,
                             plane_wave_expansion=initial_field.plane_wave_expansion(layer_system, 0)[0])
    else:  # top illumination
        i_top = layer_system.number_of_layers() - 1
        far_field = FarField(vacuum_wavelength=initial_field.vacuum_wavelength,
                             plane_wave_expansion=initial_field.plane_wave_expansion(layer_system, i_top)[1])

    return far_field


def scattered_far_field(vacuum_wavelength, particle_list, layer_system, polar_angles=None, azimuthal_angles=None):
    """
    Evaluate the scattered far field.

    Args:
        vacuum_wavelength (float):      in length units
        particle_list (list):           list of smuthi.Particle objects
        layer_system (smuthi.layers.LayerSystem): represents the stratified medium
        polar_angles (np array):        polar angles values (radian)
        azimuthal_angles (np array):    azimuthal angle values (radian)

    Returns:
        A tuple of FarField objects, one for forward scattering (i.e., into the top hemisphere) and one for backward
        scattering (bottom hemisphere).
    """
    omega = coord.angular_frequency(vacuum_wavelength)
    if polar_angles is None:
        polar_angles = np.arange(0, 181, 1, dtype=float) * np.pi / 180
    if azimuthal_angles is None:
        azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180

    i_top = layer_system.number_of_layers() - 1
    top_polar_angles = polar_angles[polar_angles <= (np.pi / 2)]
    bottom_polar_angles = polar_angles[polar_angles > (np.pi / 2)]
    neff_top = np.sort(np.sin(top_polar_angles) * layer_system.refractive_indices[i_top])
    neff_bottom = np.sort(np.sin(bottom_polar_angles) * layer_system.refractive_indices[0])
    k_top = omega * layer_system.refractive_indices[i_top]
    z_top = layer_system.reference_z(i_top)
    k_bottom = omega * layer_system.refractive_indices[0]
    z_bottom = 0
    pwe_top = fldex.PlaneWaveExpansion(k=k_top, k_parallel=neff_top*omega, azimuthal_angles=azimuthal_angles,
                                       type='upgoing', reference_point=[0,0,z_top], valid_between=(z_top, np.inf))
    pwe_bottom = fldex.PlaneWaveExpansion(k=k_bottom, k_parallel=neff_bottom*omega, azimuthal_angles=azimuthal_angles,
                                          type='downgoing', reference_point=[0,0,z_bottom],
                                          valid_between=(-np.inf,z_bottom))

    for iS, particle in enumerate(particle_list):
        i_iS = layer_system.layer_number(particle.position[2])

        # direct contribution
        if i_iS == i_top:
            pwe_up, _ = fldex.swe_to_pwe_conversion(particle.scattered_field, neff_top*omega, azimuthal_angles,
                                                    layer_system)
            pwe_top = pwe_top + pwe_up
        if i_iS == 0:
            _, pwe_down = fldex.swe_to_pwe_conversion(particle.scattered_field, neff_bottom*omega, azimuthal_angles,
                                                      layer_system)
            pwe_bottom = pwe_bottom + pwe_down

        # layer mediated contribution
        pwe_up, _ = fldex.swe_to_pwe_conversion(particle.scattered_field, neff_top*omega, azimuthal_angles,
                                                layer_system, i_top, True)
        pwe_top = pwe_top + pwe_up
        _, pwe_down = fldex.swe_to_pwe_conversion(particle.scattered_field, neff_bottom*omega, azimuthal_angles,
                                                  layer_system, 0, True)
        pwe_bottom = pwe_bottom + pwe_down

    top_far_field = FarField(vacuum_wavelength=vacuum_wavelength, plane_wave_expansion=pwe_top)
    bottom_far_field = FarField(vacuum_wavelength=vacuum_wavelength, plane_wave_expansion=pwe_bottom)

    return top_far_field, bottom_far_field


def scattering_cross_section(initial_field, particle_list, layer_system, polar_angles=None, azimuthal_angles=None):
    """Evaluate and display the differential scattering cross section as a function of solid angle.

    Args:
        initial_field (smuthi.initial.PlaneWave): Initial Plane wave
        particle_list (list):   scattering particles
        layer_system (smuthi.layers.LayerSystem): stratified medium
        polar_angles (array like):  polar angles (radian), default: from 1 to 180 degree in steps of 1
        azimuthal_angles (array like): azimuthal angles (radian), default: from 1 to 360 degree in steps of 1

    Returns:
        A dictionary with the following entries
            - 'differential':     Differential cross section as an np.array of dimension 2 x nb x na where nb is the number of polar angles and na is the number of azimuthal angles. The first index refers to polarization.
            - 'total':            Total cross section as list with two entries (for two polarizations)
            - 'polar':            Polar differential cross section (that is scattered power per polar angle, divided by incoming intensity)
            - 'polar angles':     Polar angles for which the differential cross section was evaluated
            - 'azimuthal angles': Azimuthal angles for which the differential cross section was evaluated
            - 'forward indices':  The indices of polar angles that correspond to directions in the top hemisphere
            - 'backward indices': The indices of polar angles that correspond to directions in the bottom hemisphere
    """
    if not type(initial_field).__name__ == 'PlaneWave':
        raise ValueError('Cross section only defined for plane wave excitation.')

    if polar_angles is None:
        polar_angles = (np.concatenate([np.arange(0, 90, 1, dtype=float), np.arange(91, 181, 1, dtype=float)])
                        * np.pi / 180)
    if azimuthal_angles is None:
        azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180

    i_top = layer_system.number_of_layers() - 1
    vacuum_wavelength = initial_field.vacuum_wavelength
    omega = coord.angular_frequency(vacuum_wavelength)
    k_bot = omega * layer_system.refractive_indices[0]
    k_top = omega * layer_system.refractive_indices[-1]

    # read plane wave parameters
    A_P = initial_field.amplitude
    beta_P = initial_field.polar_angle
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
    far_field = scattered_far_field(vacuum_wavelength, particle_list, layer_system, polar_angles, azimuthal_angles)
    dscs = far_field.intensity / initial_intensity

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


def extinction_cross_section(initial_field, particle_list, layer_system):
    """Evaluate and display the differential scattering cross section as a function of solid angle.

    Args:
        initial_field (smuthi.initial_field.PlaneWave): Plane wave object
        particle_list (list): List of smuthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem): Representing the stratified medium

    Returns:
        Dictionary with following entries
            - 'forward':      Extinction in the positinve z-direction (top layer)
            - 'backward':     Extinction in the negative z-direction (bottom layer)
    """
    if not type(initial_field).__name__ == 'PlaneWave':
        raise ValueError('Cross section only defined for plane wave excitation.')

    i_top = layer_system.number_of_layers() - 1
    z_top = layer_system.reference_z(i_top)
    vacuum_wavelength = initial_field.vacuum_wavelength
    omega = coord.angular_frequency(vacuum_wavelength)
    k_bot = omega * layer_system.refractive_indices[0]
    k_top = omega * layer_system.refractive_indices[-1]

    # read plane wave parameters
    pol_P = initial_field.polarization
    beta_P = initial_field.polar_angle
    alpha_P = initial_field.azimuthal_angle

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
    rvec_0 = np.array(initial_field.reference_point)
    ejkriP = np.exp(1j * np.dot(kvec_P, rvec_iP - rvec_0))
    A_P = initial_field.amplitude * ejkriP

    initial_intensity = abs(A_P) ** 2 * abs(np.cos(beta_P)) * n_P / 2

    pwe_scat_top = fldex.PlaneWaveExpansion(k=k_top, k_parallel=kappa_P, azimuthal_angles=alpha_P, type='upgoing',
                                            reference_point=[0, 0, z_top], valid_between=(z_top, np.inf))
    pwe_scat_bottom = fldex.PlaneWaveExpansion(k=k_bot, k_parallel=kappa_P, azimuthal_angles=alpha_P, type='downgoing',
                                               reference_point=[0, 0, 0], valid_between=(-np.inf, 0))

    for iS, particle in enumerate(particle_list):
        i_iS = layer_system.layer_number(particle.position[2])

        # direct contribution
        if i_iS == i_top:
            pwe_up, _ = fldex.swe_to_pwe_conversion(particle.scattered_field, kappa_P, alpha_P, layer_system)
            pwe_scat_top = pwe_scat_top + pwe_up
        if i_iS == 0:
            _, pwe_down = fldex.swe_to_pwe_conversion(particle.scattered_field, kappa_P, alpha_P, layer_system)
            pwe_scat_bottom = pwe_scat_bottom + pwe_down

        # layer mediated contribution
        pwe_up, _ = fldex.swe_to_pwe_conversion(particle.scattered_field, kappa_P, alpha_P, layer_system, i_top, True)
        pwe_scat_top = pwe_scat_top + pwe_up
        _, pwe_down = fldex.swe_to_pwe_conversion(particle.scattered_field, kappa_P, alpha_P, layer_system, 0, True)
        pwe_scat_bottom = pwe_scat_bottom + pwe_down

    # bottom extinction
    _, pwe_init_bottom = initial_field.plane_wave_expansion(layer_system, 0)
    kz_bot = coord.k_z(k_parallel=kappa_P, k=k_bot)
    gRPbot =  np.squeeze(pwe_init_bottom.coefficients[pol_P])
    g_scat_bottom = np.squeeze(pwe_scat_bottom.coefficients[pol_P])
    P_bot_ext = 4 * np.pi ** 2 * kz_bot / omega * (gRPbot * np.conj(g_scat_bottom)).real
    bottom_extinction_cs = - P_bot_ext / initial_intensity

    # top extinction
    pwe_init_top, _ = initial_field.plane_wave_expansion(layer_system, i_top)
    gRPtop = np.squeeze(pwe_init_top.coefficients[pol_P])
    kz_top = coord.k_z(k_parallel=kappa_P, k=k_top)
    g_scat_top = np.squeeze(pwe_scat_top.coefficients[pol_P])
    P_top_ext = 4 * np.pi ** 2 * kz_top / omega * (gRPtop * np.conj(g_scat_top)).real
    top_extinction_cs = - P_top_ext / initial_intensity

    extinction_cs = {'top': top_extinction_cs, 'bottom': bottom_extinction_cs}

    return extinction_cs


