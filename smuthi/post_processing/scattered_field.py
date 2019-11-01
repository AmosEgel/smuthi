"""Manage post processing steps to evaluate the scattered near and far field"""
import numpy as np
import smuthi.coordinates as coord
import smuthi.field_expansion as fldex
from tqdm import tqdm
import sys


def total_far_field(initial_field, particle_list, layer_system, polar_angles='default', azimuthal_angles='default'):
    """
    Evaluate the total far field, the initial far field and the scattered far field. Cannot be used if initial field
    is a plane wave.
    
    Args:
        initial_field (smuthi.initial_field.InitialField): represents the initial field
        particle_list (list):                       list of smuthi.Particle objects
        layer_system (smuthi.layers.LayerSystem):   represents the stratified medium
        polar_angles (numpy.ndarray or str):        polar angles values (radian). 
                                                    if 'default', use smuthi.coordinates.default_polar_angles
        azimuthal_angles (numpy.ndarray or str):    azimuthal angle values (radian)
                                                    if 'default', use smuthi.coordinates.default_azimuthal_angles

    Returns:
        A tuple of three smuthi.field_expansion.FarField objects for total, initial and scattered far field. Mind that the scattered far field
        has no physical meaning and is for illustration purposes only. 
    """ 
    if not (type(initial_field).__name__ == 'GaussianBeam' or type(initial_field).__name__ == 'DipoleSource'
            or type(initial_field).__name__ == 'DipoleCollection'):
        raise ValueError('only for Gaussian beams and dipole sources')
    omega = initial_field.angular_frequency()
    vacuum_wavelength = initial_field.vacuum_wavelength
    if type(polar_angles) == str and polar_angles == 'default':
        polar_angles = coord.default_polar_angles
    if type(azimuthal_angles) == str and azimuthal_angles == 'default':
        azimuthal_angles = coord.default_azimuthal_angles

    if any(polar_angles.imag):
        raise ValueError("complex angles not allowed in far field")

    i_top = layer_system.number_of_layers() - 1
    top_polar_angles = polar_angles[polar_angles < (np.pi / 2)]
    bottom_polar_angles = polar_angles[polar_angles > (np.pi / 2)]
    neff_top = np.sort(np.sin(top_polar_angles) * layer_system.refractive_indices[i_top])
    neff_bottom = np.sort(np.sin(bottom_polar_angles) * layer_system.refractive_indices[0])
    
    if len(top_polar_angles) > 1 and layer_system.refractive_indices[i_top].imag == 0:
        pwe_scat_top, _ = scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, i_top,
                                              k_parallel=neff_top*omega, azimuthal_angles=azimuthal_angles,
                                              include_direct=True, include_layer_response=True)
        pwe_in_top, _ = initial_field.plane_wave_expansion(layer_system, i_top, k_parallel_array=neff_top*omega,
                                                           azimuthal_angles_array=azimuthal_angles)
        pwe_top = pwe_scat_top + pwe_in_top
        top_far_field = fldex.pwe_to_ff_conversion(vacuum_wavelength=vacuum_wavelength, plane_wave_expansion=pwe_top)
        top_far_field_init = fldex.pwe_to_ff_conversion(vacuum_wavelength=vacuum_wavelength, 
                                                        plane_wave_expansion=pwe_in_top)
        top_far_field_scat = fldex.pwe_to_ff_conversion(vacuum_wavelength=vacuum_wavelength, 
                                                        plane_wave_expansion=pwe_scat_top)
    else:
        top_far_field = None
        top_far_field_init = None
        top_far_field_scat = None

    if len(bottom_polar_angles) > 1 and layer_system.refractive_indices[0].imag == 0:
        _, pwe_scat_bottom = scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, 0,
                                                 k_parallel=neff_bottom*omega, azimuthal_angles=azimuthal_angles,
                                                 include_direct=True, include_layer_response=True)
        _, pwe_in_bottom = initial_field.plane_wave_expansion(layer_system, 0, k_parallel_array=neff_bottom*omega,
                                                              azimuthal_angles_array=azimuthal_angles)
        pwe_bottom = pwe_scat_bottom + pwe_in_bottom
        bottom_far_field = fldex.pwe_to_ff_conversion(vacuum_wavelength=vacuum_wavelength,
                                                      plane_wave_expansion=pwe_bottom)
        bottom_far_field_init = fldex.pwe_to_ff_conversion(vacuum_wavelength=vacuum_wavelength,
                                                           plane_wave_expansion=pwe_in_bottom)
        bottom_far_field_scat = fldex.pwe_to_ff_conversion(vacuum_wavelength=vacuum_wavelength,
                                                           plane_wave_expansion=pwe_scat_bottom)
    else:
        bottom_far_field = None
        bottom_far_field_init = None
        bottom_far_field_scat = None

    if top_far_field is not None:
        far_field = top_far_field
        far_field_init = top_far_field_init
        far_field_scat = top_far_field_scat
        if bottom_far_field is not None:
            far_field.append(bottom_far_field)
            far_field_init.append(bottom_far_field_init)
            far_field_scat.append(bottom_far_field_scat)
    else:
        far_field = bottom_far_field
        far_field_init = bottom_far_field_init
        far_field_scat = bottom_far_field_scat

    far_field.polar_angles = far_field.polar_angles.real
    far_field_init.polar_angles = far_field_init.polar_angles.real
    far_field_scat.polar_angles = far_field_scat.polar_angles.real
    return far_field, far_field_init, far_field_scat


def scattered_far_field(vacuum_wavelength, particle_list, layer_system, polar_angles='default', 
                        azimuthal_angles='default'):
    """
    Evaluate the scattered far field.

    Args:
        vacuum_wavelength (float):                  in length units
        particle_list (list):                       list of smuthi.Particle objects
        layer_system (smuthi.layers.LayerSystem):   represents the stratified medium
        polar_angles (numpy.ndarray or str):        polar angles values (radian). 
                                                    if 'default', use smuthi.coordinates.default_polar_angles
        azimuthal_angles (numpy.ndarray or str):    azimuthal angle values (radian)
                                                    if 'default', use smuthi.coordinates.default_azimuthal_angles

    Returns:
        A smuthi.field_expansion.FarField object of the scattered field.
    """
    omega = coord.angular_frequency(vacuum_wavelength)
    if type(polar_angles) == str and polar_angles == 'default':
        polar_angles = coord.default_polar_angles
    if type(azimuthal_angles) == str and azimuthal_angles == 'default':
        azimuthal_angles = coord.default_azimuthal_angles

    if any(polar_angles.imag):
        raise ValueError("complex angles not allowed in far field")

    i_top = layer_system.number_of_layers() - 1
    top_polar_angles = polar_angles[polar_angles < (np.pi / 2)]
    bottom_polar_angles = polar_angles[polar_angles > (np.pi / 2)]
    neff_top = np.sort(np.sin(top_polar_angles) * layer_system.refractive_indices[i_top])
    neff_bottom = np.sort(np.sin(bottom_polar_angles) * layer_system.refractive_indices[0])

    if len(top_polar_angles) > 1 and layer_system.refractive_indices[i_top].imag == 0:
        pwe_top, _ = scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, i_top,
                                         k_parallel=neff_top*omega, azimuthal_angles=azimuthal_angles,
                                         include_direct=True, include_layer_response=True)
        top_far_field = fldex.pwe_to_ff_conversion(vacuum_wavelength=vacuum_wavelength, plane_wave_expansion=pwe_top)
    else:
        top_far_field = None

    if len(bottom_polar_angles) > 1 and layer_system.refractive_indices[0].imag == 0:
        _, pwe_bottom = scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, 0,
                                            k_parallel=neff_bottom*omega, azimuthal_angles=azimuthal_angles,
                                            include_direct=True, include_layer_response=True)
        bottom_far_field = fldex.pwe_to_ff_conversion(vacuum_wavelength=vacuum_wavelength,
                                                      plane_wave_expansion=pwe_bottom)
    else:
        bottom_far_field = None

    if top_far_field is not None:
        far_field = top_far_field
        if bottom_far_field is not None:
            far_field.append(bottom_far_field)
    else:
        far_field = bottom_far_field

    far_field.polar_angles = far_field.polar_angles.real
    return far_field


def scattering_cross_section(initial_field, particle_list, layer_system, polar_angles='default', 
                             azimuthal_angles='default'):
    """Evaluate and display the differential scattering cross section as a function of solid angle.

    Args:
        initial_field (smuthi.initial.PlaneWave): Initial Plane wave
        particle_list (list):                     scattering particles
        layer_system (smuthi.layers.LayerSystem): stratified medium
        polar_angles (numpy.ndarray or str):        polar angles values (radian). 
                                                    if 'default', use smuthi.coordinates.default_polar_angles
        azimuthal_angles (numpy.ndarray or str):    azimuthal angle values (radian)
                                                    if 'default', use smuthi.coordinates.default_azimuthal_angles

    Returns:
        A tuple of smuthi.field_expansion.FarField objects, one for forward scattering (i.e., into the top hemisphere) and one for backward
        scattering (bottom hemisphere).
    """
    if not type(initial_field).__name__ == 'PlaneWave':
        raise ValueError('Cross section only defined for plane wave excitation.')

    i_top = layer_system.number_of_layers() - 1
    vacuum_wavelength = initial_field.vacuum_wavelength
    omega = coord.angular_frequency(vacuum_wavelength)
    k_bot = omega * layer_system.refractive_indices[0]
    k_top = omega * layer_system.refractive_indices[-1]

    # read plane wave parameters
    A_P = initial_field.amplitude
    beta_P = initial_field.polar_angle
    if np.cos(beta_P) > 0:
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

    dscs = scattered_far_field(vacuum_wavelength, particle_list, layer_system, polar_angles, azimuthal_angles)
    dscs.signal_type = 'differential scattering cross section'
    dscs.signal = dscs.signal / initial_intensity

    return dscs


def extinction_cross_section(initial_field, particle_list, layer_system):
    """Evaluate the extinction cross section.

    Args:
        initial_field (smuthi.initial_field.PlaneWave): Plane wave object
        particle_list (list): List of smuthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem): Representing the stratified medium

    Returns:
        Dictionary with following entries
            - 'top':      Extinction in the positinve z-direction (top layer)
            - 'bottom':     Extinction in the negative z-direction (bottom layer)
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

    if np.cos(beta_P) > 0:
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

    pwe_scat_top, _ = scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, i_top, kappa_P, alpha_P)

    _, pwe_scat_bottom = scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, 0, kappa_P, alpha_P)

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


def scattered_field_piecewise_expansion(vacuum_wavelength, particle_list, layer_system, k_parallel='default', 
                                        azimuthal_angles='default', layer_numbers=None):
    """Compute a piecewise field expansion of the scattered field.

    Args:
        vacuum_wavelength (float):                  vacuum wavelength
        particle_list (list):                       list of smuthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem):   stratified medium
        k_parallel (numpy.ndarray or str):          in-plane wavenumbers array. 
                                                    if 'default', use smuthi.coordinates.default_k_parallel
        azimuthal_angles (numpy.ndarray or str):    azimuthal angles array
                                                    if 'default', use smuthi.coordinates.default_azimuthal_angles
        layer_numbers (list):                       if specified, append only plane wave expansions for these layers
        

    Returns:
        scattered field as smuthi.field_expansion.PiecewiseFieldExpansion object

    """
    
    if layer_numbers is None:
        layer_numbers = range(layer_system.number_of_layers())
        
    sfld = fldex.PiecewiseFieldExpansion()
    for i in tqdm(layer_numbers, desc='Scatt. field expansion    ', file=sys.stdout,
                                        bar_format='{l_bar}{bar}| elapsed: {elapsed} ' 'remaining: {remaining}'):
        # layer mediated scattered field ---------------------------------------------------------------------------
        k = coord.angular_frequency(vacuum_wavelength) * layer_system.refractive_indices[i]
        ref = [0, 0, layer_system.reference_z(i)]
        vb = (layer_system.lower_zlimit(i), layer_system.upper_zlimit(i))
        pwe_up = fldex.PlaneWaveExpansion(k=k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles, kind='upgoing',
                                          reference_point=ref, lower_z=vb[0], upper_z=vb[1])
        pwe_down = fldex.PlaneWaveExpansion(k=k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles,
                                            kind='downgoing', reference_point=ref, lower_z=vb[0], upper_z=vb[1])
        for particle in particle_list:
            add_up, add_down = fldex.swe_to_pwe_conversion(particle.scattered_field, k_parallel, azimuthal_angles,
                                                           layer_system, i, True)
            pwe_up = pwe_up + add_up
            pwe_down = pwe_down + add_down

        # in bottom_layer, suppress upgoing waves, and in top layer, suppress downgoing waves
        if i > 0:
            sfld.expansion_list.append(pwe_up)
        if i < layer_system.number_of_layers()-1:
            sfld.expansion_list.append(pwe_down)

    # direct field ---------------------------------------------------------------------------------------------
    for particle in particle_list:
        sfld.expansion_list.append(particle.scattered_field)

    return sfld


def scattered_field_pwe(vacuum_wavelength, particle_list, layer_system, layer_number, k_parallel='default',
                        azimuthal_angles='default', include_direct=True, include_layer_response=True):
    """Calculate the plane wave expansion of the scattered field of a set of particles.

    Args:
        vacuum_wavelength (float):          Vacuum wavelength (length unit)
        particle_list (list):               List of Particle objects
        layer_system (smuthi.layers.LayerSystem):  Stratified medium
        layer_number (int):                 Layer number in which the plane wave expansion should be valid
        k_parallel (numpy.ndarray or str):          in-plane wavenumbers array. 
                                                    if 'default', use smuthi.coordinates.default_k_parallel
        azimuthal_angles (numpy.ndarray or str):    azimuthal angles array
                                                    if 'default', use smuthi.coordinates.default_azimuthal_angles
        include_direct (bool):              If True, include the direct scattered field
        include_layer_response (bool):      If True, include the layer system response

    Returns:
        A tuple of PlaneWaveExpansion objects for upgoing and downgoing waves.
    """

    sys.stdout.write('Evaluating scattered field plane wave expansion in layer number %i ...\n'%layer_number)
    sys.stdout.flush()

    omega = coord.angular_frequency(vacuum_wavelength)
    k = omega * layer_system.refractive_indices[layer_number]
    z = layer_system.reference_z(layer_number)
    vb = (layer_system.lower_zlimit(layer_number), layer_system.upper_zlimit(layer_number))
    pwe_up = fldex.PlaneWaveExpansion(k=k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles, kind='upgoing',
                                      reference_point=[0, 0, z], lower_z=vb[0], upper_z=vb[1])
    pwe_down = fldex.PlaneWaveExpansion(k=k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles, kind='downgoing',
                                        reference_point=[0, 0, z], lower_z=vb[0], upper_z=vb[1])

    for iS, particle in enumerate(tqdm(particle_list, desc='Scatt. field pwe          ', file=sys.stdout,
                                        bar_format='{l_bar}{bar}| elapsed: {elapsed} ' 'remaining: {remaining}')):

        i_iS = layer_system.layer_number(particle.position[2])

        # direct contribution
        if i_iS == layer_number and include_direct:
            pu, pd = fldex.swe_to_pwe_conversion(swe=particle.scattered_field, k_parallel=k_parallel,
                                                 azimuthal_angles=azimuthal_angles, layer_system=layer_system)
            pwe_up = pwe_up + pu
            pwe_down = pwe_down + pd

        # layer mediated contribution
        if include_layer_response:
            pu, pd = fldex.swe_to_pwe_conversion(swe=particle.scattered_field, k_parallel=k_parallel,
                                                 azimuthal_angles=azimuthal_angles, layer_system=layer_system,
                                                 layer_number=layer_number, layer_system_mediated=True)
            pwe_up = pwe_up + pu
            pwe_down = pwe_down + pd

    return pwe_up, pwe_down
