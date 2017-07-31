# -*- coding: utf-8 -*-

import numpy as np
import smuthi.coordinates as coord
import smuthi.layers as lay
import smuthi.field_expansion as fldex


class InitialField:
    """Base class for initial field classes"""
    def __init__(self, vacuum_wavelength):
        self.vacuum_wavelength = vacuum_wavelength

    def spherical_wave_expansion(self, particle, layer_system):
        """Virtual method to be overwritten."""
        pass

    def plane_wave_expansion(self, layer_system):
        """Virtual method to be overwritten."""
        pass


class PlaneWave(InitialField):
    """Class for the representation of a plane wave as initial field.

    Args:
        vacuum_wavelength (float):
        polar_angle (float):            polar angle of k-vector (0 means, k is parallel to z-axis)
        azimuthal_angle (float):        azimuthal angle of k-vector (0 means, k is in x-z plane)
        polarization (int):             0 for TE/s, 1 for TM/p
        amplitude (float or complex):   Plane wave amplitude at reference point
        reference_point (list):         Location where electric field of incoming wave equals amplitude
    """
    def __init__(self, vacuum_wavelength, polar_angle, azimuthal_angle, polarization, amplitude=1,
                 reference_point=None):
        InitialField.__init__(self, vacuum_wavelength)
        self.polar_angle = polar_angle
        self.azimuthal_angle = azimuthal_angle
        self.polarization = polarization
        self.amplitude = amplitude
        if reference_point:
            self.reference_point = reference_point
        else:
            self.reference_point = [0, 0, 0]

    def plane_wave_expansion(self, layer_system, z):
        """Plane wave expansion for the plane wave including its layer system response. As it already is a plane wave,
        the plane wave expansion is somehow trivial (containing only one partial wave, i.e., a discrete plane wave
        expansion).

        Args:
            layer_system (smuthi.layers.LayerSystem): Layer system object
            z (float): position at which the PWE should be valid

        Returns:
            Tuple of smuthi.field_expansion.PlaneWaveExpansion objects. The first element is an upgoing PWE, whereas the
            second element is a downgoing PWE.
        """
        if np.cos(self.polar_angle) > 0:
            iP = 0
            ud_P = 0  # 0 for upwards
            type = 'upgoing'
        else:
            iP = layer_system.number_of_layers() - 1
            ud_P = 1  # 1 for downwards
            type = 'downgoing'

        niP = layer_system.refractive_indices[iP]
        neff = np.sin([self.polar_angle]) * niP
        alpha = np.array([self.azimuthal_angle])

        angular_frequency = coord.angular_frequency(self.vacuum_wavelength)
        k_iP = niP * angular_frequency
        k_Px = k_iP * np.sin(self.polar_angle) * np.cos(self.azimuthal_angle)
        k_Py = k_iP * np.sin(self.polar_angle) * np.sin(self.azimuthal_angle)
        k_Pz = k_iP * np.cos(self.polar_angle)
        z_iP = layer_system.reference_z(iP)
        amplitude_iP = self.amplitude * np.exp(-1j * (k_Px * self.reference_point[0] + k_Py * self.reference_point[1]
                                                   + k_Pz * (self.reference_point[2] - z_iP)))
        iP_between = (layer_system.lower_zlimit(iP), layer_system.upper_zlimit(iP))
        pwe_exc = fldex.PlaneWaveExpansion(k=k_iP, k_parallel=neff*angular_frequency, azimuthal_angles=alpha, type=type,
                                           reference_point=[0, 0, z_iP], valid_between=iP_between)
        pwe_exc.coefficients[self.polarization, 0, 0] = amplitude_iP
        iz = layer_system.layer_number(z)
        pwe_up, pwe_down = layer_system.response(pwe_exc, from_layer=iP, to_layer=iz)
        if iP == iz:
            if type == 'upgoing':
                pwe_up = pwe_up + pwe_exc
            elif type == 'downgoing':
                pwe_down = pwe_down + pwe_exc

        return pwe_up, pwe_down

    def spherical_wave_expansion(self, particle, layer_system):
        """Regular spherical wave expansion of the plane wave including layer system response, at the locations of the
        particles
        """
        pwe_up, pwe_down = self.plane_wave_expansion(layer_system, particle.position[2])
        return (fldex.pwe_to_swe_conversion(pwe_up, particle.l_max, particle.m_max, particle.position)
                + fldex.pwe_to_swe_conversion(pwe_down, particle.l_max, particle.m_max, particle.position))
