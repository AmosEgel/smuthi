# -*- coding: utf-8 -*-

import numpy as np
import smuthi.coordinates as coord
import smuthi.layers as lay
import smuthi.vector_wave_functions as vwf
import smuthi.field_expansion as fldex


class InitialField:
    """Base class for initial field classes"""
    def __init__(self, vacuum_wavelength):
        self.vacuum_wavelength = vacuum_wavelength

    def spherical_wave_expansion(self, particle_collection, layer_system):
        """Virtual method to be overwritten."""
        pass

    def plane_wave_expansion(self, layer_system, layer_number=None):
        """Virtual method to be overwritten."""
        pass


class PlaneWave(InitialField):
    """Class for the representation of a plane wave as initial field."""
    def __init__(self, vacuum_wavelength, polar_angle, azimuthal_angle, polarization, amplitude=1,
                 reference_point=None):
        InitialField.__init__(self, vacuum_wavelength)
        self.polar_angle = polar_angle
        self.azimuthal_angle = azimuthal_angle
        self.polarization = polarization
        self.amplitude = amplitude
        self.reference_point = reference_point

    def spherical_wave_expansion(self, particle_collection, layer_system):




        def planewave_swe_coefficients(vacuum_wavelength=None, amplitude=1, polar_angle=0, azimuthal_angle=0,
                                       polarization=0,
                                       planewave_reference_point=[0, 0, 0], particle_position=[0, 0, 0],
                                       layer_system=None,
                                       layerresponse_precision=None):
            """Return the initial field coefficients (spherical wave expansion) as a numpy-array for a single particle in a
            planarly layered medium and a single initial plane wave. The coefficients array has dimension NS x nmax, where NS is
            the number of particles and nmax is the number of swe coefficients per particle.

            Input:
            vacuum_wavelength           (length unit)
            amplitude                   (default=1)
            polar_angle                 The polar angle also decides if the plane wave comes from the bottom side of the layer
                                        system (polar angle<=pi/2) or from the top side (radian, default=0)
            azimuthal_angle             (radian, default=0)
            polarization                (0=TE, 1=TM)
            planewave_reference_point   In the format [x,y,z]. At this point in space the amplitude of the incoming wave is as
                                        defined. (length unit, default=[0,0,0])
            particle_position           In the format [x,y,z]. (length unit, default=[0,0,0])
            layer_system                smuthi.layers.LayerSystem object that defines the planarly layered medium in which the
                                        particle is located
            layerresponse_precision     If None, standard numpy is used for the layer response. If int>0, that many decimal
                                        digits are considered in multiple precision. (default=None)
            """

            l_max = idx.l_max
            m_max = idx.m_max

            angular_frequency = coord.angular_frequency(vacuum_wavelength)
            blocksize = idx.number_of_indices()

            # initialize output
            aPR = np.zeros(blocksize, dtype=complex)  # layer system mediated
            aPD = np.zeros(blocksize, dtype=complex)  # direct

            if polar_angle < (np.pi / 2):  # then the plane wave comes from the bottom layer
                iP = 0
            else:  # top layer
                iP = layer_system.number_of_layers() - 1

            # wavevectors in initial layer iP and in particle layer iS
            k_iP = layer_system.refractive_indices[iP] * angular_frequency
            kp = k_iP * np.sin(polar_angle)
            kx = np.cos(azimuthal_angle) * kp
            ky = np.sin(azimuthal_angle) * kp
            pm_kz_iP = k_iP * np.cos(polar_angle)
            kvec_iP = np.array([kx, ky, pm_kz_iP])

            iS = layer_system.layer_number(particle_position[2])
            k_iS = layer_system.refractive_indices[iS] * angular_frequency
            kz_iS = coord.k_z(k_parallel=kp, k=k_iS)

            kvec_pl_iS = np.array([kx, ky, kz_iS])
            kvec_mn_iS = np.array([kx, ky, -kz_iS])

            rvec_iP = np.array([0, 0, layer_system.reference_z(iP)])
            rvec_iS = np.array([0, 0, layer_system.reference_z(iS)])
            rvec_S = np.array(particle_position)
            rvec_0 = np.array(planewave_reference_point)

            # phase factors
            ejkriP = np.exp(1j * np.dot(kvec_iP, rvec_iP - rvec_0))
            ejkplriSS = np.exp(1j * np.dot(kvec_pl_iS, rvec_S - rvec_iS))
            ejkmnriSS = np.exp(1j * np.dot(kvec_mn_iS, rvec_S - rvec_iS))
            ejkriPS = np.exp(1j * np.dot(kvec_iP, rvec_S))

            L = lay.layersystem_response_matrix(pol=polarization, layer_d=layer_system.thicknesses,
                                                layer_n=layer_system.refractive_indices, kpar=kp,
                                                omega=angular_frequency,
                                                fromlayer=iP, tolayer=layer_system.layer_number(particle_position[2]),
                                                precision=layerresponse_precision)
            gR = np.dot(L, np.array([1, 1]))
            for tau in range(2):
                for m in range(-m_max, m_max + 1):
                    emjma = np.exp(-1j * m * azimuthal_angle)
                    for l in range(max(1, abs(m)), l_max + 1):
                        n = idx.multi_to_single_index(tau, l, m)
                        Bdagpl = vwf.transformation_coefficients_VWF(tau, l, m, pol=polarization, kp=kp, kz=kz_iS,
                                                                     dagger=True)
                        Bdagmn = vwf.transformation_coefficients_VWF(tau, l, m, pol=polarization, kp=kp, kz=-kz_iS,
                                                                     dagger=True)
                        Bvec = np.array([Bdagpl * ejkplriSS, Bdagmn * ejkmnriSS])
                        aPR[n] = 4 * amplitude * emjma * ejkriP * np.dot(Bvec, gR)
                        if iS == iP:  # add direct contribution
                            Bin = vwf.transformation_coefficients_VWF(tau, l, m, pol=polarization, kp=kp, kz=pm_kz_iP,
                                                                      dagger=True)
                            aPD[n] = 4 * amplitude * emjma * ejkriPS * Bin

            return aPR + aPD




class InitialFieldCollection:
    """Collection of initial field parameter sets."""
    def __init__(self, vacuum_wavelength=None):
        """Initialize

        input:
        vacuum_wavelength   (length unit)
        """
        self.vacuum_wavelength = vacuum_wavelength

        # A list of dictionaries, each corresponding to one element of the initial field. For example, a simulation with
        # N dipole sources would have N entries in the list
        self.specs_list = []

    def add_planewave(self, amplitude=1, polar_angle=None, reference_point=[0, 0, 0], azimuthal_angle=0,
                      polarization=0, k_parallel=0, n_effective=None, layer=None):
        """Add plane wave to the collection of initial field components.

        Redundant entries (like k_parallel AND n_effective) don't need to be specified.

        input:
        amplitude:          complex amplitude of plane wave at x,y,z=0,0,0
        k_parallel:         in-plane wavenumber (inverse length)
        n_effective:        k_parallel = omega * n_effective
        polar_angle:        polar angle in radians
        refractive_index:   refractive index of medium from which plane wave comes
        azimuthal_angle:    azimuthal angle in radians (that is the angle between in-plane k-vector and x-axis)
        polarization:       polarization (0='TE' and E~e_azimuth, 1='TM' and E~e_polar).
        layer:              'top' for top illumination and 'bottom' for bottom illumination
        """
        if polarization == 'TE':
            polarization = 0
        elif polarization == 'TM':
            polarization = 1

        if polar_angle is None:
            if n_effective is None:
                n_effective = k_parallel / coord.angular_frequency(self.vacuum_wavelength)
            if layer == 'top':
                polar_angle = np.pi - np.asin(n_effective)
            elif layer == 'bottom':
                polar_angle = np.asin(n_effective)

        self.specs_list.append({'type': 'plane wave', 'amplitude': amplitude, 'polar angle': polar_angle,
                                'azimuthal angle': azimuthal_angle, 'polarization': polarization,
                                'reference point': reference_point})


def initial_field_swe_coefficients(initial_field_collection, particle_collection, layer_system,
                                   layerresponse_precision=None):
    """Return the spherical wave expansion coefficients a0 for all particles specified in a
    smuthi.particles.ParticleCollection object, for all initial field elements. a0 is a numpy array of dimensions
    NS x nmax, where NS is the number of particles and nmax is the number of swe coefficients per particle."""
    a0 = np.zeros((particle_collection.particle_number(), idx.number_of_indices()), dtype=complex)
    for iS, rS in enumerate(particle_collection.particle_positions()):
        for i0, specs in enumerate(initial_field_collection.specs_list):
            if specs['type'] == 'plane wave':
                a0[iS, :] += planewave_swe_coefficients(vacuum_wavelength=initial_field_collection.vacuum_wavelength,
                                                        amplitude=specs['amplitude'],
                                                        polar_angle=specs['polar angle'],
                                                        azimuthal_angle=specs['azimuthal angle'],
                                                        polarization=specs['polarization'],
                                                        planewave_reference_point=specs['reference point'],
                                                        particle_position=rS, layer_system=layer_system,
                                                        layerresponse_precision=layerresponse_precision)
            else:
                raise ValueError('This initial field type is currently not implemented')

    return a0


def planewave_swe_coefficients(vacuum_wavelength=None, amplitude=1, polar_angle=0, azimuthal_angle=0, polarization=0,
                               planewave_reference_point=[0, 0, 0], particle_position=[0, 0, 0], layer_system=None,
                               layerresponse_precision=None):
    """Return the initial field coefficients (spherical wave expansion) as a numpy-array for a single particle in a
    planarly layered medium and a single initial plane wave. The coefficients array has dimension NS x nmax, where NS is
    the number of particles and nmax is the number of swe coefficients per particle.

    Input:
    vacuum_wavelength           (length unit)
    amplitude                   (default=1)
    polar_angle                 The polar angle also decides if the plane wave comes from the bottom side of the layer
                                system (polar angle<=pi/2) or from the top side (radian, default=0)
    azimuthal_angle             (radian, default=0)
    polarization                (0=TE, 1=TM)
    planewave_reference_point   In the format [x,y,z]. At this point in space the amplitude of the incoming wave is as
                                defined. (length unit, default=[0,0,0])
    particle_position           In the format [x,y,z]. (length unit, default=[0,0,0])
    layer_system                smuthi.layers.LayerSystem object that defines the planarly layered medium in which the
                                particle is located
    layerresponse_precision     If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """

    l_max = idx.l_max
    m_max = idx.m_max

    angular_frequency = coord.angular_frequency(vacuum_wavelength)
    blocksize = idx.number_of_indices()
    
    #initialize output
    aPR = np.zeros(blocksize, dtype=complex) # layer system mediated
    aPD = np.zeros(blocksize, dtype=complex) # direct
    
    if polar_angle < (np.pi / 2): #then the plane wave comes from the bottom layer 
        iP = 0
    else: #top layer
        iP = layer_system.number_of_layers() - 1
    
    #wavevectors in initial layer iP and in particle layer iS
    k_iP = layer_system.refractive_indices[iP] * angular_frequency
    kp = k_iP * np.sin(polar_angle)
    kx = np.cos(azimuthal_angle) * kp
    ky = np.sin(azimuthal_angle) * kp
    pm_kz_iP = k_iP * np.cos(polar_angle)
    kvec_iP = np.array([kx, ky, pm_kz_iP])

    iS = layer_system.layer_number(particle_position[2])
    k_iS = layer_system.refractive_indices[iS] * angular_frequency
    kz_iS = coord.k_z(k_parallel=kp, k=k_iS)

    kvec_pl_iS = np.array([kx, ky, kz_iS])
    kvec_mn_iS = np.array([kx, ky, -kz_iS])

    rvec_iP = np.array([0, 0, layer_system.reference_z(iP)])
    rvec_iS = np.array([0, 0, layer_system.reference_z(iS)])
    rvec_S = np.array(particle_position)
    rvec_0 = np.array(planewave_reference_point)

    # phase factors
    ejkriP = np.exp(1j * np.dot(kvec_iP, rvec_iP - rvec_0))
    ejkplriSS = np.exp(1j * np.dot(kvec_pl_iS, rvec_S - rvec_iS))
    ejkmnriSS = np.exp(1j * np.dot(kvec_mn_iS, rvec_S - rvec_iS))
    ejkriPS = np.exp(1j * np.dot(kvec_iP, rvec_S))

    L = lay.layersystem_response_matrix(pol=polarization, layer_d=layer_system.thicknesses,
                                        layer_n=layer_system.refractive_indices, kpar=kp, omega=angular_frequency,
                                        fromlayer=iP, tolayer=layer_system.layer_number(particle_position[2]),
                                        precision=layerresponse_precision)
    gR = np.dot(L, np.array([1, 1]))
    for tau in range(2):
        for m in range(-m_max, m_max + 1):
            emjma = np.exp(-1j * m * azimuthal_angle)
            for l in range(max(1, abs(m)), l_max+1):
                n = idx.multi_to_single_index(tau, l, m)
                Bdagpl = vwf.transformation_coefficients_VWF(tau, l, m, pol=polarization, kp=kp, kz=kz_iS, dagger=True)
                Bdagmn = vwf.transformation_coefficients_VWF(tau, l, m, pol=polarization, kp=kp, kz=-kz_iS, dagger=True)
                Bvec = np.array([Bdagpl * ejkplriSS, Bdagmn * ejkmnriSS])
                aPR[n] = 4 * amplitude * emjma * ejkriP * np.dot(Bvec, gR)
                if iS == iP: # add direct contribution
                    Bin = vwf.transformation_coefficients_VWF(tau, l, m, pol=polarization, kp=kp, kz=pm_kz_iP,
                                                              dagger=True)
                    aPD[n] = 4 * amplitude * emjma * ejkriPS * Bin

    return aPR + aPD

