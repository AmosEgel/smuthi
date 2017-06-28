# -*- coding: utf-8 -*-

import numpy as np
import smuthi.coordinates as coord
import smuthi.layers as lay
import smuthi.field_expansion as fldex


class InitialField:
    """Base class for initial field classes"""
    def __init__(self, vacuum_wavelength):
        self.vacuum_wavelength = vacuum_wavelength

    def spherical_wave_expansion(self, particle_collection, layer_system):
        """Virtual method to be overwritten."""
        pass

    def plane_wave_expansion(self, layer_system):
        """Virtual method to be overwritten."""
        pass


class PlaneWave(InitialField):
    """Class for the representation of a plane wave as initial field.

    .. todo:: testing
    """
    def __init__(self, vacuum_wavelength, polar_angle, azimuthal_angle, polarization, amplitude=1,
                 reference_point=None):
        InitialField.__init__(self, vacuum_wavelength)
        self.polar_angle = polar_angle
        self.azimuthal_angle = azimuthal_angle
        self.polarization = polarization
        self.amplitude = amplitude
        self.reference_point = reference_point

    def plane_wave_expansion(self, layer_system):

        if np.cos(self.polar_angle) > 0:
            iP = 0
            ud_P = 0  # 0 for upwards
        else:
            iP = layer_system.number_of_layers - 1
            ud_P = 1  # 1 for downwards

        niP = layer_system.refractive_indices[iP]
        neff = np.sin([self.polar_angle]) * niP
        alpha = np.array([self.azimuthal_angle])

        if self.reference_point:
            angular_frequency = coord.angular_frequency(self.vacuum_wavelength)
            k_iP = niP * angular_frequency
            k_Px = k_iP * np.sin(self.polar_angle) * np.cos(self.azimuthal_angle)
            k_Py = k_iP * np.sin(self.polar_angle) * np.sin(self.azimuthal_angle)
            k_Pz = k_iP * np.cos(self.polar_angle)
            z_iP = layer_system.reference_z(iP)
            amplitude = self.amplitude * np.exp(-1j * (k_Px * self.reference_point[0] + k_Py * self.reference_point[1]
                                                       + k_Pz * (self.reference_point[2] - z_iP)))
        else:
            amplitude = self.amplitude

        gexc = fldex.PlaneWaveExpansion(neff, alpha, layer_system)
        gexc.coefficients[iP][self.polarization, ud_P, 0, 0] = self.amplitude
        gR = gexc.response(vacuum_wavelength=self.vacuum_wavelength, excitation_layer_number=iP)
        gtotal = gexc + gR

        return gtotal

    def spherical_wave_expansion(self, particle_collection, layer_system):
        gtotal = self.plane_wave_expansion(layer_system)
        a = gtotal.spherical_wave_expansion(self.vacuum_wavelength, particle_collection)
        return a
