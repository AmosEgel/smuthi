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
        polar_angles (numpy.ndarray):       Polar angles (default: from 0 to 180 degree in steps of 1 degree)
        azimuthal_angles (numpy.ndarray):   Azimuthal angles (default: from 0 to 360 degree in steps of 1 degree)
        vacuum_wavelength (float):          Vacuum wavelength
        plane_wave_expansion (smuthi.field_expansion.PlaneWaveExpansion):   Plane wave expansion for which the far field
                                                                            power is computed
    """
    def __init__(self, polar_angles=None, azimuthal_angles=None, vacuum_wavelength=None, plane_wave_expansion=None,
                 type='intensity'):
        if plane_wave_expansion is None:
            if polar_angles is None:
                polar_angles = np.arange(0, 181, 1, dtype=float) * np.pi / 180
            if azimuthal_angles is None:
                azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180
            self.polar_angles = polar_angles
            self.azimuthal_angles = azimuthal_angles

            # The far field signal is represented as a 3-dimensional numpy.ndarray.
            # The indices are:
            # - polarization (0=TE, 1=TM)
            # - index of the polar angle
            # - index of the azimuthal angle
            self.signal = np.zeros(2, polar_angles.__len__, azimuthal_angles.__len__)  # dimensions: j, beta, alpha
            self.signal.fill(np.nan)
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

            if any(self.polar_angles.imag):
                raise ValueError('complex angles are not allowed')

            self.azimuthal_angles = plane_wave_expansion.azimuthal_angles

            kkz2 = coord.k_z(k_parallel=kp, k=k) ** 2 * k
            self.signal = (2 * np.pi ** 2 / omega * kkz2[np.newaxis, :, np.newaxis]
                           * abs(plane_wave_expansion.coefficients) ** 2).real
        self.type = type

    def azimuthal_integral(self):
        if len(self.azimuthal_angles) > 2:
            return np.trapz(self.signal, self.azimuthal_angles[None, None, :]) * np.sin(self.polar_angles[None, :])
        else:
            return None

    def integral(self):
        if len(self.azimuthal_angles) > 2:
            return np.trapz(self.azimuthal_integral(), self.polar_angles[None, :])
        else:
            return None


def scattered_far_field(vacuum_wavelength, particle_list, layer_system, polar_angles=None, azimuthal_angles=None):
    """
    Evaluate the scattered far field.

    Args:
        vacuum_wavelength (float):                  in length units
        particle_list (list):                       list of smuthi.Particle objects
        layer_system (smuthi.layers.LayerSystem):   represents the stratified medium
        polar_angles (numpy.ndarray):               polar angles values (radian)
        azimuthal_angles (numpy.ndarray):           azimuthal angle values (radian)

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

    if top_polar_angles.__len__() > 1 and layer_system.refractive_indices[i_top].imag == 0:
        pwe_top, _ = fldex.scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, i_top,
                                               k_parallel=neff_top*omega, azimuthal_angles=azimuthal_angles,
                                               include_direct=True, include_layer_response=True)
        top_far_field = FarField(vacuum_wavelength=vacuum_wavelength, plane_wave_expansion=pwe_top)
    else:
        top_far_field = None

    if bottom_polar_angles.__len__() > 1 and layer_system.refractive_indices[0].imag == 0:
        _, pwe_bottom = fldex.scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, 0,
                                                  k_parallel=neff_bottom*omega, azimuthal_angles=azimuthal_angles,
                                                  include_direct=True, include_layer_response=True)
        bottom_far_field = FarField(vacuum_wavelength=vacuum_wavelength, plane_wave_expansion=pwe_bottom)
    else:
        bottom_far_field = None

    return top_far_field, bottom_far_field


def scattering_cross_section(initial_field, particle_list, layer_system, polar_angles=None, azimuthal_angles=None):
    """Evaluate and display the differential scattering cross section as a function of solid angle.

    Args:
        initial_field (smuthi.initial.PlaneWave): Initial Plane wave
        particle_list (list):                     scattering particles
        layer_system (smuthi.layers.LayerSystem): stratified medium
        polar_angles (numpy.ndarray):             polar angles (radian), default: from 1 to 180 degree in steps of 1
        azimuthal_angles (numpy.ndarray):         azimuthal angles (radian), default: from 1 to 360 degree in steps of 1

    Returns:
        A tuple of FarField objects, one for forward scattering (i.e., into the top hemisphere) and one for backward
        scattering (bottom hemisphere).
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

    top_dscs, bottom_dscs = scattered_far_field(vacuum_wavelength, particle_list, layer_system, polar_angles,
                                                azimuthal_angles)
    if top_dscs:
        top_dscs.type = 'differential scattering cross section'
        top_dscs.signal = top_dscs.signal / initial_intensity
    if bottom_dscs:
        bottom_dscs.type = 'differential scattering cross section'
        bottom_dscs.signal = bottom_dscs.signal / initial_intensity

    return top_dscs, bottom_dscs


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

    pwe_scat_top, _ = fldex.scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, i_top, kappa_P, alpha_P)

    _, pwe_scat_bottom = fldex.scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, 0, kappa_P, alpha_P)

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


class NearField:
    def __init__(self, x, y, z):
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.electric_field_x = np.zeros(x.shape, dtype=complex)
        self.electric_field_y = np.zeros(x.shape, dtype=complex)
        self.electric_field_z = np.zeros(x.shape, dtype=complex)


def scattered_electric_field(x, y, z, k_parallel, azimuthal_angles, vacuum_wavelength, particle_list, layer_system):
    """Complex electric scattered near field.

    Args:
        x (numpy array):    x-coordinates of points in space where to evaluate field.
        y (numpy array):    y-coordinates of points in space where to evaluate field.
        z (numpy array):    z-coordinates of points in space where to evaluate field.
        k_parallel (numpy.ndarray):        In plane wavenumbers for the plane wave expansion
        azimuthal_angles (numpy.ndarray):   Azimuthal angles for the plane wave expansion
        vacuum_wavelength (float):          Vacuum wavelength
        particle_list (list):               List of smuthi.particle.Particle objects
        layer_system (smuthi.layers.LayerSystem):   Stratified medium
    Returns:
        NearField object holding the scattered electric field.
    """

    near_field = NearField(x, y, z)

    old_shape = x.shape
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    electric_field_x = np.zeros(x.shape, dtype=complex)
    electric_field_y = np.zeros(x.shape, dtype=complex)
    electric_field_z = np.zeros(x.shape, dtype=complex)

    layer_numbers = []
    for zi in z:
        layer_numbers.append(layer_system.layer_number(zi))

    for i in range(layer_system.number_of_layers()):
        layer_indices = [ii for ii, laynum in enumerate(layer_numbers) if laynum == i]
        if layer_indices:

            # layer mediated scattered field ---------------------------------------------------------------------------
            k = coord.angular_frequency(vacuum_wavelength) * layer_system.refractive_indices[i]
            ref = [0, 0, layer_system.reference_z(i)]
            vb = (layer_system.lower_zlimit(i), layer_system.upper_zlimit(i))
            pwe_up = fldex.PlaneWaveExpansion(k=k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles,
                                              type='upgoing', reference_point=ref, valid_between=vb)
            pwe_down = fldex.PlaneWaveExpansion(k=k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles,
                                                type='downgoing', reference_point=ref, valid_between=vb)
            for particle in particle_list:
                add_up, add_down = fldex.swe_to_pwe_conversion(particle.scattered_field, k_parallel, azimuthal_angles,
                                                               layer_system, i, True)
                pwe_up = pwe_up + add_up
                pwe_down = pwe_down + add_down
            ex_up, ey_up, ez_up = pwe_up.electric_field(x[layer_indices], y[layer_indices], z[layer_indices])
            ex_down, ey_down, ez_down = pwe_down.electric_field(x[layer_indices], y[layer_indices], z[layer_indices])
            electric_field_x[layer_indices] = ex_up + ex_down
            electric_field_y[layer_indices] = ey_up + ey_down
            electric_field_z[layer_indices] = ez_up + ez_down

            # direct field ---------------------------------------------------------------------------------------------
            for particle in particle_list:
                if layer_system.layer_number(particle.position[2]) == i:
                    ex, ey, ez = particle.scattered_field.electric_field(x[layer_indices], y[layer_indices],
                                                                         z[layer_indices])
                    electric_field_x[layer_indices] += ex
                    electric_field_y[layer_indices] += ey
                    electric_field_z[layer_indices] += ez
                    # todo:check if swe valid, fill in NaN or something otherwise

    near_field.electric_field_x = electric_field_x.reshape(old_shape)
    near_field.electric_field_y = electric_field_y.reshape(old_shape)
    near_field.electric_field_z = electric_field_z.reshape(old_shape)

    return near_field
