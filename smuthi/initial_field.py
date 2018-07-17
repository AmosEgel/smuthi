# -*- coding: utf-8 -*-

import numpy as np
import smuthi.coordinates as coord
import smuthi.field_expansion as fldex
import smuthi.vector_wave_functions as vwf
import smuthi.particles as part
import smuthi.particle_coupling as pc
import smuthi.scattered_field as sf
import smuthi.memoizing as memo
import warnings
import sys
from tqdm import tqdm


class InitialField:
    """Base class for initial field classes"""
    def __init__(self, vacuum_wavelength):
        self.vacuum_wavelength = vacuum_wavelength

    def spherical_wave_expansion(self, particle, layer_system):
        """Virtual method to be overwritten."""
        pass

    def plane_wave_expansion(self, layer_system, i):
        """Virtual method to be overwritten."""
        pass

    def piecewise_field_expansion(self, layer_system):
        """Virtual method to be overwritten."""
        pass
    
    def angular_frequency(self):
        """Angular frequency.
        
        Returns:
            Angular frequency (float) according to the vacuum wavelength in units of c=1.
        """
        return coord.angular_frequency(self.vacuum_wavelength)


class InitialPropagatingWave(InitialField):
    """Base class for plane waves and Gaussian beams

    Args:
        vacuum_wavelength (float):
        polar_angle (float):            polar propagation angle (0 means, parallel to z-axis)
        azimuthal_angle (float):        azimuthal propagation angle (0 means, in x-z plane)
        polarization (int):             0 for TE/s, 1 for TM/p
        amplitude (float or complex):   Electric field amplitude
        reference_point (list):         Location where electric field of incoming wave equals amplitude
    """
    def __init__(self, vacuum_wavelength, polar_angle, azimuthal_angle, polarization, amplitude=1,
                 reference_point=None):
        assert (polarization == 0 or polarization == 1)
        InitialField.__init__(self, vacuum_wavelength)
        if np.isclose(np.cos(polar_angle), 0):
            raise ValueError('propagating waves not defined in the xy-plane')
        self.polar_angle = polar_angle
        self.azimuthal_angle = azimuthal_angle
        self.polarization = polarization
        self.amplitude = amplitude
        if reference_point:
            self.reference_point = reference_point
        else:
            self.reference_point = [0, 0, 0]
            
    def spherical_wave_expansion(self, particle, layer_system):
        """Regular spherical wave expansion of the wave including layer system response, at the locations of the
        particles.
        
        Args:
            particle (smuthi.particles.Particle):    particle relative to which the swe is computed
            layer_system (smuthi.layer.LayerSystem): stratified medium
            
        Returns:
            regular smuthi.field_expansion.SphericalWaveExpansion object
        """
        i = layer_system.layer_number(particle.position[2])
        pwe_up, pwe_down = self.plane_wave_expansion(layer_system, i)
        return (fldex.pwe_to_swe_conversion(pwe_up, particle.l_max, particle.m_max, particle.position)
                + fldex.pwe_to_swe_conversion(pwe_down, particle.l_max, particle.m_max, particle.position))

    def piecewise_field_expansion(self, layer_system):
        """Compute a piecewise field expansion of the initial field.
        
        Args:
            layer_system (smuthi.layer.LayerSystem):    stratified medium
            
        Returns:
            smuthi.field_expansion.PiecewiseWaveExpansion object
        """
        pfe = fldex.PiecewiseFieldExpansion()
        for i in range(layer_system.number_of_layers()):
            pwe_up, pwe_down = self.plane_wave_expansion(layer_system, i)
            pfe.expansion_list.append(pwe_up)
            pfe.expansion_list.append(pwe_down)
        return pfe

    def electric_field(self, x, y, z, layer_system):
        """Evaluate the complex electric field corresponding to the wave.

        Args:
            x (array like):     Array of x-values where to evaluate the field (length unit)
            y (array like):     Array of y-values where to evaluate the field (length unit)
            z (array like):     Array of z-values where to evaluate the field (length unit)
            layer_system (smuthi.layer.LayerSystem):    Stratified medium

        Returns
            Tuple (E_x, E_y, E_z) of electric field values
        """

        pfe = self.piecewise_field_expansion(layer_system=layer_system)
        return pfe.electric_field(x, y, z)

    
class GaussianBeam(InitialPropagatingWave):
    """Class for the representation of a Gaussian beam as initial field."""
    def __init__(self, vacuum_wavelength, polar_angle, azimuthal_angle, polarization, beam_waist, 
                 k_parallel_array='default', azimuthal_angles_array='default', amplitude=1, reference_point=None):
        InitialPropagatingWave.__init__(self, vacuum_wavelength, polar_angle, azimuthal_angle, polarization, amplitude,
                                        reference_point)
        self.beam_waist = beam_waist
        if type(k_parallel_array) == str and k_parallel_array == 'default':
            k_parallel_array = coord.default_k_parallel
        if type(azimuthal_angles_array) == str and azimuthal_angles_array == 'default':
            azimuthal_angles_array = coord.default_azimuthal_angles
        self.k_parallel_array = k_parallel_array
        self.azimuthal_angles_array = azimuthal_angles_array
        
    def plane_wave_expansion(self, layer_system, i, k_parallel_array=None, azimuthal_angles_array=None):
        """Plane wave expansion of the Gaussian beam.
        
        Args:
            layer_system (smuthi.layer.LayerSystem):    stratified medium
            i (int):                                    layer number in which to evaluate the expansion
            k_parallel_array (numpy.ndarray):           in-plane wavenumber array for the expansion. if none specified,
                                                        self.k_parallel_array is used
            azimuthal_angles_array (numpy.ndarray):     azimuthal angles for the expansion. if none specified,
                                                        self.azimuthal_angles_array is used
            
        Returns:
            tuple of to smuthi.field_expansion.PlaneWaveExpansion objects, one for upgoing and one for downgoing 
            component
        """            
        if k_parallel_array is None:
            k_parallel_array = self.k_parallel_array
        if azimuthal_angles_array is None:
            azimuthal_angles_array = self.azimuthal_angles_array

        if np.cos(self.polar_angle) > 0:
            iG = 0  # excitation layer number
            kind = 'upgoing'
        else:
            iG = layer_system.number_of_layers() - 1
            kind = 'downgoing'

        niG = layer_system.refractive_indices[iG]  # refractive index in excitation layer
        if niG.imag:
            warnings.warn('beam coming from absorbing medium')
        k_iG = niG * self.angular_frequency()
        z_iG = layer_system.reference_z(iG)
        loz = layer_system.lower_zlimit(iG)
        upz = layer_system.upper_zlimit(iG)
        pwe_exc = fldex.PlaneWaveExpansion(k=k_iG, k_parallel=k_parallel_array, azimuthal_angles=azimuthal_angles_array, 
                                           kind=kind, reference_point=[0, 0, z_iG], lower_z=loz, upper_z=upz)

        k_Gx = k_iG * np.sin(self.polar_angle) * np.cos(self.azimuthal_angle)
        k_Gy = k_iG * np.sin(self.polar_angle) * np.sin(self.azimuthal_angle)

        kp = pwe_exc.k_parallel_grid()
        al = pwe_exc.azimuthal_angle_grid()

        kx = kp * np.cos(al)
        ky = kp * np.sin(al)
        kz = pwe_exc.k_z_grid()

        w = self.beam_waist
        r_G = self.reference_point

        g = (self.amplitude * w**2 / (4 * np.pi) * np.exp(-w**2 / 4 * ((kx - k_Gx)**2 + (ky - k_Gy)**2))
             * np.exp(-1j * (kx * r_G[0] + ky * r_G[1] + kz * (r_G[2] - z_iG))) )

        pwe_exc.coefficients[0, :, :] = g * np.cos(al - self.azimuthal_angle + self.polarization * np.pi/2)
        if np.cos(self.polar_angle) > 0:
            pwe_exc.coefficients[1, :, :] = g * np.sin(al - self.azimuthal_angle + self.polarization * np.pi/2)
        else:
            pwe_exc.coefficients[1, :, :] = - g * np.sin(al - self.azimuthal_angle + self.polarization * np.pi/2)

        pwe_up, pwe_down = layer_system.response(pwe_exc, from_layer=iG, to_layer=i)
        if iG == i:
            if kind == 'upgoing':
                pwe_up = pwe_up + pwe_exc
            elif kind == 'downgoing':
                pwe_down = pwe_down + pwe_exc

        return pwe_up, pwe_down

    def propagated_far_field(self, layer_system):
        """Evaluate the far field intensity of the reflected / transmitted initial field.

        Args:
            layer_system (smuthi.layers.LayerSystem):           Stratified medium

        Returns:
            A tuple of smuthi.field_expansion.FarField objects, one for forward (i.e., into the top hemisphere) and one
            for backward propagation (bottom hemisphere).
        """
        i_top = layer_system.number_of_layers() - 1
        top_ff = fldex.pwe_to_ff_conversion(vacuum_wavelength=self.vacuum_wavelength,
                                            plane_wave_expansion=self.plane_wave_expansion(layer_system, i_top)[0])
        bottom_ff = fldex.pwe_to_ff_conversion(vacuum_wavelength=self.vacuum_wavelength,
                                               plane_wave_expansion=self.plane_wave_expansion(layer_system, 0)[1])

        return top_ff, bottom_ff

    def initial_intensity(self, layer_system):
        """Evaluate the incoming intensity of the initial field.

        Args:
            layer_system (smuthi.layers.LayerSystem):           Stratified medium

        Returns:
            A smuthi.field_expansion.FarField object holding the initial intensity information.
        """
        if np.cos(self.polar_angle) > 0:  # bottom illumination
            ff = fldex.pwe_to_ff_conversion(vacuum_wavelength=self.vacuum_wavelength,
                                            plane_wave_expansion=self.plane_wave_expansion(layer_system, 0)[0])
        else:  # top illumination
            i_top = layer_system.number_of_layers() - 1
            ff = fldex.pwe_to_ff_conversion(vacuum_wavelength=self.vacuum_wavelength,
                                            plane_wave_expansion=self.plane_wave_expansion(layer_system, i_top)[1])
        return ff


class PlaneWave(InitialPropagatingWave):
    """Class for the representation of a plane wave as initial field.

    Args:
        vacuum_wavelength (float):
        polar_angle (float):            polar angle of k-vector (0 means, k is parallel to z-axis)
        azimuthal_angle (float):        azimuthal angle of k-vector (0 means, k is in x-z plane)
        polarization (int):             0 for TE/s, 1 for TM/p
        amplitude (float or complex):   Plane wave amplitude at reference point
        reference_point (list):         Location where electric field of incoming wave equals amplitude
    """
        
    def plane_wave_expansion(self, layer_system, i):
        """Plane wave expansion for the plane wave including its layer system response. As it already is a plane wave,
        the plane wave expansion is somehow trivial (containing only one partial wave, i.e., a discrete plane wave
        expansion).

        Args:
            layer_system (smuthi.layers.LayerSystem): Layer system object
            i (int): layer number in which the plane wave expansion is valid

        Returns:
            Tuple of smuthi.field_expansion.PlaneWaveExpansion objects. The first element is an upgoing PWE, whereas the
            second element is a downgoing PWE.
        """
        if np.cos(self.polar_angle) > 0:
            iP = 0
            kind = 'upgoing'
        else:
            iP = layer_system.number_of_layers() - 1
            kind = 'downgoing'

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
        loz = layer_system.lower_zlimit(iP)
        upz = layer_system.upper_zlimit(iP)
        pwe_exc = fldex.PlaneWaveExpansion(k=k_iP, k_parallel=neff*angular_frequency, azimuthal_angles=alpha, kind=kind,
                                           reference_point=[0, 0, z_iP], lower_z=loz, upper_z=upz)
        pwe_exc.coefficients[self.polarization, 0, 0] = amplitude_iP
        pwe_up, pwe_down = layer_system.response(pwe_exc, from_layer=iP, to_layer=i)
        if iP == i:
            if kind == 'upgoing':
                pwe_up = pwe_up + pwe_exc
            elif kind == 'downgoing':
                pwe_down = pwe_down + pwe_exc

        return pwe_up, pwe_down


class DipoleSource(InitialField):
    """Class for the representation of a single point dipole source.

    Args:
        vacuum_wavelength (float):      vacuum wavelength (length units)
        dipole_moment (list or tuple):  (x, y, z)-coordinates of dipole moment vector
        position (list or tuple):       (x, y, z)-coordinates of dipole position
        k_parallel (numpy.ndarray or str):          In-plane wavenumber. 
                                                    If 'default', use smuthi.coordinates.default_k_parallel
        azimuthal_angles (numpy.ndarray or str):    Azimuthal angles for plane wave expansions
                                                    If 'default', use smuthi.coordinates.default_azimuthal_angles
    """
    def __init__(self, vacuum_wavelength, dipole_moment, position, k_parallel='default', azimuthal_angles='default'):
        InitialField.__init__(self, vacuum_wavelength)
        self.dipole_moment = dipole_moment
        self.position = position
        self.k_parallel = k_parallel
        self.azimuthal_angles = azimuthal_angles

    def current(self):
        r"""The current density takes the form

        .. math::
            \mathbf{j}(\mathbf{r}) = \delta(\mathbf{r} - \mathbf{r}_D) \mathbf{j}_D,

        where :math:`\mathbf{j}_D = -j \omega \mathbf{\mu}`, :math:`\mathbf{r}_D` is the location of the dipole, :math:`\omega`
        is the angular frequency and :math:`\mathbf{\mu}` is the dipole moment.
        For further details, see 'Principles of nano optics' by Novotny and Hecht.

        Returns:
            List of [x, y, z]-components of current density vector :math:`\mathbf{j}_D`
        """
        return [- 1j * self.angular_frequency() * self.dipole_moment[i] for i in range(3)]

    def outgoing_spherical_wave_expansion(self, layer_system):
        """The dipole field as an expansion in spherical vector wave functions.

        Args:
            layer_system (smuthi.layers.LayerSystem):   stratified medium

        Returns:
            outgoing smuthi.field_expansion.SphericalWaveExpansion object
        """
        laynum = layer_system.layer_number(self.position[2])
        k = layer_system.refractive_indices[laynum] * self.angular_frequency()
        swe_out = fldex.SphericalWaveExpansion(k=k, l_max=1, m_max=1, kind='outgoing', reference_point=self.position,
                                               lower_z=layer_system.lower_zlimit(laynum),
                                               upper_z=layer_system.upper_zlimit(laynum))
        l = 1
        for tau in range(2):
            for m in range(-1, 2):
                ex, ey, ez = vwf.spherical_vector_wave_function(0, 0, 0, k, 1, tau, l, -m)
                b = 1j * k / np.pi * 1j * self.angular_frequency() * (ex * self.current()[0] + ey * self.current()[1]
                                                                      + ez * self.current()[2])
                swe_out.coefficients[fldex.multi_to_single_index(tau, l, m, 1, 1)] = b

        return swe_out

    def spherical_wave_expansion(self, particle, layer_system):
        """Regular spherical wave expansion of the wave including layer system response, at the locations of the
        particles.

        Args:
            particle (smuthi.particles.Particle):    particle relative to which the swe is computed
            layer_system (smuthi.layer.LayerSystem): stratified medium

        Returns:
            regular smuthi.field_expansion.SphericalWaveExpansion object
        """
        virtual_particle = part.Particle(position=self.position, l_max=1, m_max=1)
        wd = pc.direct_coupling_block(vacuum_wavelength=self.vacuum_wavelength, receiving_particle=particle,
                                      emitting_particle=virtual_particle, layer_system=layer_system)
        wr = pc.layer_mediated_coupling_block(vacuum_wavelength=self.vacuum_wavelength, receiving_particle=particle,
                                              emitting_particle=virtual_particle, layer_system=layer_system,
                                              k_parallel=self.k_parallel)
        k = self.angular_frequency() * layer_system.refractive_indices[layer_system.layer_number(particle.position[2])]
        swe = fldex.SphericalWaveExpansion(k=k, l_max=particle.l_max, m_max=particle.m_max, kind='regular',
                                           reference_point=particle.position)
        swe.coefficients = np.dot(wd + wr, self.outgoing_spherical_wave_expansion(layer_system).coefficients)
        return swe

    def piecewise_field_expansion(self, layer_system, include_direct_field=True, include_layer_response=True):
        """Compute a piecewise field expansion of the dipole field.

        Args:
            layer_system (smuthi.layer.LayerSystem):    stratified medium
            include_direct_field (bool):                if True (default), the direct dipole field is included.
                                                        otherwise, only the layer response of the dipole field is
                                                        returned.
            include_layer_response (bool):              if True (default), the layer response of the dipole field is 
                                                        included. otherwise, only the direct dipole field is
                                                        returned.
                                        

        Returns:
            smuthi.field_expansion.PiecewiseWaveExpansion object
        """
        pfe = fldex.PiecewiseFieldExpansion()
        if include_direct_field:
            pfe.expansion_list.append(self.outgoing_spherical_wave_expansion(layer_system))
        if include_layer_response:
            for i in range(layer_system.number_of_layers()):
                # layer response as plane wave expansions
                pwe_up, pwe_down = fldex.swe_to_pwe_conversion(swe=self.outgoing_spherical_wave_expansion(layer_system),
                                                            k_parallel=self.k_parallel,
                                                            azimuthal_angles=self.azimuthal_angles,
                                                            layer_system=layer_system, layer_number=i,
                                                            layer_system_mediated=True)
                if i > 0:
                    pfe.expansion_list.append(pwe_up)
                if i < layer_system.number_of_layers() - 1:
                    pfe.expansion_list.append(pwe_down)

        return pfe

    def electric_field(self, x, y, z, layer_system, include_direct_field=True, include_layer_response=True):
        """Evaluate the complex electric field of the dipole source.

        Args:
            x (array like):     Array of x-values where to evaluate the field (length unit)
            y (array like):     Array of y-values where to evaluate the field (length unit)
            z (array like):     Array of z-values where to evaluate the field (length unit)
            layer_system (smuthi.layer.LayerSystem):    Stratified medium
            include_direct_field (bool):                if True (default), the direct dipole field is included.
                                                        otherwise, only the layer response of the dipole field is
                                                        returned.
            include_layer_response (bool):              if True (default), the layer response of the dipole field is 
                                                        included. otherwise, only the direct dipole field is
                                                        returned.


        Returns
            Tuple (E_x, E_y, E_z) of electric field values
        """
        pfe = self.piecewise_field_expansion(layer_system=layer_system, include_direct_field=include_direct_field,
                                             include_layer_response=include_layer_response)
        return pfe.electric_field(x, y, z)

    def dissipated_power_homogeneous_background(self, layer_system):
        r"""Compute the power that the dipole would radiate in an infinite homogeneous medium of the same refractive
        index as the layer that contains the dipole.

        .. math::
            P_0 = \frac{|\mathbf{\mu}| k \omega^3}{12 \pi}

        Args:
            layer_system (smuthi.layers.LayerSystem): stratified medium

        Returns:
            power (float)
        """
        laynum = layer_system.layer_number(self.position[2])
        k = layer_system.refractive_indices[laynum] * self.angular_frequency()
        mu2 = abs(self.dipole_moment[0])**2 + abs(self.dipole_moment[1])**2 + abs(self.dipole_moment[2])**2
        p = mu2 * k * self.angular_frequency()**3 / (12 * np.pi)
        return p

    def check_dissipated_power_homogeneous_background(self, layer_system):
        laynum = layer_system.layer_number(self.position[2])
        e_x_in, e_y_in, e_z_in = self.electric_field(x=self.position[0]+10, y=self.position[1]+10, z=self.position[2]+10,
                                                     layer_system=layer_system, include_direct_field=True)
        k = layer_system.refractive_indices[laynum] * self.angular_frequency()
        p = self.angular_frequency() / 2 * (np.conjugate(self.dipole_moment[0]) * (e_x_in)
                                            + np.conjugate(self.dipole_moment[1]) * (e_y_in)
                                            + np.conjugate(self.dipole_moment[2]) * (e_z_in)).imag
        return p

    def dissipated_power(self, particle_list, layer_system, show_progress=True):
        r"""Compute the power that the dipole feeds into the system.

        It is computed according to

        .. math::
            P = P_0 + \frac{\omega}{2} \mathrm{Im} (\mathbf{\mu}^* \cdot \mathbf{E}(\mathbf{r}_D))

        where :math:`P_0` is the power that the dipole would feed into an infinte homogeneous medium with the same
        refractive index as the layer that contains the dipole, :math:`\mathbf{r}_D` is the location of the dipole,
        :math:`\omega` is the angular frequency, :math:`\mathbf{\mu}` is the dipole moment and :math:`\mathbf{E}`
        includes the reflections of the dipole field from the layer interfaces, as well as the scattered field from all
        particles.

        Args:
            particle_list (list of smuthi.particles.Particle objects): scattering particles
            layer_system (smuthi.layers.LayerSystem):                  stratified medium
            show_progress (bool):                                      if true, display progress
            
        Returns:
            dissipated power as float
        """

        k = layer_system.wavenumber(layer_system.layer_number(self.position[2]), self.vacuum_wavelength)
        virtual_particle = part.Particle(position=self.position, l_max=1, m_max=1)
        scattered_field_swe = fldex.SphericalWaveExpansion(k=k, l_max=1, m_max=1, kind='regular',
                                                           reference_point=self.position)
        
        if show_progress:
            iterator = tqdm(particle_list, desc='Dipole dissipated power   ', file=sys.stdout,
                            bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}')
        else:
            iterator = particle_list
                    
        for particle in iterator:
            wd = pc.direct_coupling_block(vacuum_wavelength=self.vacuum_wavelength,
                                        receiving_particle=virtual_particle,
                                        emitting_particle=particle,
                                        layer_system=layer_system)

            wr = pc.layer_mediated_coupling_block(vacuum_wavelength=self.vacuum_wavelength,
                                                receiving_particle=virtual_particle,
                                                emitting_particle=particle,
                                                layer_system=layer_system,
                                                k_parallel=self.k_parallel)

            scattered_field_swe.coefficients += np.dot(wd + wr, particle.scattered_field.coefficients)

        e_x_scat, e_y_scat, e_z_scat = scattered_field_swe.electric_field(x=self.position[0],
                                                                          y=self.position[1],
                                                                          z=self.position[2])

        e_x_in, e_y_in, e_z_in = self.electric_field(x=self.position[0], y=self.position[1], z=self.position[2],
                                                     layer_system=layer_system, include_direct_field=False)

        power = self.angular_frequency() / 2 * (np.conjugate(self.dipole_moment[0]) * (e_x_scat + e_x_in)
                                                + np.conjugate(self.dipole_moment[1]) * (e_y_scat + e_y_in)
                                                + np.conjugate(self.dipole_moment[2]) * (e_z_scat + e_z_in)).imag
        return self.dissipated_power_homogeneous_background(layer_system) + power

    def dissipated_power_alternative(self, particle_list, layer_system):
        r"""Compute the power that the dipole feeds into the system.

        It is computed according to

        .. math::
            P = P_0 + \frac{\omega}{2} \mathrm{Im} (\mathbf{\mu}^* \cdot \mathbf{E}(\mathbf{r}_D))

        where :math:`P_0` is the power that the dipole would feed into an infinte homogeneous medium with the same
        refractive index as the layer that contains the dipole, :math:`\mathbf{r}_D` is the location of the dipole,
        :math:`\omega` is the angular frequency, :math:`\mathbf{\mu}` is the dipole moment and :math:`\mathbf{E}`
        includes the reflections of the dipole field from the layer interfaces, as well as the scattered field from all
        particles. In contrast to dissipated_power, this routine relies on the scattered field piecewise expansion and
        and might thus be slower.

        Args:
            particle_list (list of smuthi.particles.Particle objects): scattering particles
            layer_system (smuthi.layers.LayerSystem): stratified medium

        Returns:
            dissipated power as float
        """
        scat_fld_exp = sf.scattered_field_piecewise_expansion(self.vacuum_wavelength, particle_list, layer_system, 
                                                              self.k_parallel, self.azimuthal_angles)
        e_x_scat, e_y_scat, e_z_scat = scat_fld_exp.electric_field(self.position[0], self.position[1], self.position[2])
        e_x_in, e_y_in, e_z_in = self.electric_field(x=self.position[0], y=self.position[1], z=self.position[2],
                                                     layer_system=layer_system, include_direct_field=False)
        power = self.angular_frequency() / 2 * (np.conjugate(self.dipole_moment[0]) * (e_x_scat + e_x_in)
                                                + np.conjugate(self.dipole_moment[1]) * (e_y_scat + e_y_in)
                                                + np.conjugate(self.dipole_moment[2]) * (e_z_scat + e_z_in)).imag
        return self.dissipated_power_homogeneous_background(layer_system) + power

    def plane_wave_expansion(self, layer_system, i, k_parallel_array=None, azimuthal_angles_array=None):
        """Plane wave expansion of the dipole field.

        Args:
            layer_system (smuthi.layer.LayerSystem):    stratified medium
            i (int):                                    layer number in which to evaluate the expansion
            k_parallel_array (numpy.ndarray):           in-plane wavenumber array for the expansion. if none specified,
                                                        self.k_parallel_array is used
            azimuthal_angles_array (numpy.ndarray):     azimuthal angles for the expansion. if none specified,
                                                        self.azimuthal_angles_array is used

        Returns:
            tuple of to smuthi.field_expansion.PlaneWaveExpansion objects, one for upgoing and one for downgoing
            component
        """
        if k_parallel_array is None:
            k_parallel_array = self.k_parallel * self.angular_frequency()
        if azimuthal_angles_array is None:
            azimuthal_angles_array = self.azimuthal_angles

        virtual_particle = part.Particle(position=self.position)
        virtual_particle.scattered_field = self.outgoing_spherical_wave_expansion(layer_system)

        return sf.scattered_field_pwe(self.vacuum_wavelength, [virtual_particle], layer_system, i, k_parallel_array,
                                      azimuthal_angles_array, include_direct=True, include_layer_response=True)


class DipoleCollection(InitialField):
    """Class for the representation of a set of point dipole sources. Use the append method to add DipoleSource objects.

    Args:
        vacuum_wavelength (float):      vacuum wavelength (length units)
        k_parallel_array (numpy.ndarray or str):          In-plane wavenumber. 
                                                          If 'default', use smuthi.coordinates.default_k_parallel
        azimuthal_angles_array (numpy.ndarray or str):    Azimuthal angles for plane wave expansions
                                                          If 'default', use smuthi.coordinates.default_azimuthal_angles
        compute_swe_by_pwe (bool):    If True, the initial field coefficients are computed through a plane wave 
                                      expansion of the whole dipole collection field. This is slower for few dipoles
                                      and particles, but can become faster than the default for many dipoles and 
                                      particles (default=False).
        compute_dissipated_power_by_pwe (bool): If True, evaluate dissipated power through a plane wave expansion of the 
                                                whole scattered field. This is slower for few dipoles,  but can be 
                                                faster than the default for many dipoles (default=False).                       
    """
    def __init__(self, vacuum_wavelength, k_parallel_array='default', azimuthal_angles_array='default', 
                 compute_swe_by_pwe=False, compute_dissipated_power_by_pwe=False):
        InitialField.__init__(self, vacuum_wavelength=vacuum_wavelength)
        self.dipole_list = []
        self.compute_swe_by_pwe = compute_swe_by_pwe
        self.compute_dissipated_power_by_pwe = compute_dissipated_power_by_pwe
        self.k_parallel_array = k_parallel_array
        self.azimuthal_angles_array = azimuthal_angles_array        

    def append(self, dipole):
        """Add dipole do collection.
        
        Args:
            dipole (DipoleSource):    Dipole object to add.
        """
        assert dipole.vacuum_wavelength == self.vacuum_wavelength
        self.dipole_list.append(dipole)

    def spherical_wave_expansion(self, particle, layer_system):
        """Regular spherical wave expansion of the dipole collection including layer system response, at the locations 
        of the particles. If self.compute_swe_by_pwe is True, use the dipole collection plane wave expansion, otherwise
        use the individual dipoles spherical_wave_expansion method.

        Args:
            particle (smuthi.particles.Particle):    particle relative to which the swe is computed
            layer_system (smuthi.layer.LayerSystem): stratified medium

        Returns:
            regular smuthi.field_expansion.SphericalWaveExpansion object
        """
        # compute by pwe?
        if self.compute_swe_by_pwe and not any(
            layer_system.layer_number(particle.position[2]) 
            == np.array([layer_system.layer_number(dip.position[2]) for dip in self.dipole_list])):
            # (make sure the particle is not in the same layer as one of the dipoles)
            return InitialPropagatingWave.spherical_wave_expansion(self, particle, layer_system)
        
        # otherwise, compute by swe of virtual particles
        k = self.angular_frequency() * layer_system.refractive_indices[layer_system.layer_number(particle.position[2])]
        swe = fldex.SphericalWaveExpansion(k=k, l_max=particle.l_max, m_max=particle.m_max, kind='regular',
                                           reference_point=particle.position)
        for dipole in self.dipole_list:
            swe = swe + dipole.spherical_wave_expansion(particle, layer_system)
        return swe
    
    def piecewise_field_expansion(self, layer_system):
        """Compute a piecewise field expansion of the dipole collection..

        Args:
            layer_system (smuthi.layer.LayerSystem):    stratified medium

        Returns:
            smuthi.field_expansion.PiecewiseWaveExpansion object
        """
        pfe = fldex.PiecewiseFieldExpansion()
        for dipole in self.dipole_list:
            pfe = pfe + dipole.piecewise_field_expansion(layer_system, include_direct_field=True)

        return pfe

    def electric_field(self, x, y, z, layer_system):
        """Evaluate the complex electric field of the dipole collection.

        Args:
            x (array like):     Array of x-values where to evaluate the field (length unit)
            y (array like):     Array of y-values where to evaluate the field (length unit)
            z (array like):     Array of z-values where to evaluate the field (length unit)
            layer_system (smuthi.layer.LayerSystem):    Stratified medium

        Returns
            Tuple (E_x, E_y, E_z) of electric field values
        """
        pfe = self.piecewise_field_expansion(layer_system=layer_system)
        return pfe.electric_field(x, y, z)

    def dissipated_power(self, particle_list, layer_system, k_parallel='default', azimuthal_angles='default'):
        r"""Compute the power that the dipole collection feeds into the system.

        It is computed according to

        .. math::
            P = \sum_i P_{0, i} + \frac{\omega}{2} \mathrm{Im} (\mathbf{\mu}_i^* \cdot \mathbf{E}_i(\mathbf{r}_i))

        where :math:`P_{0,i}` is the power that the i-th dipole would feed into an infinte homogeneous medium with the
        same refractive index as the layer that contains that dipole, :math:`\mathbf{r}_i` is the location of the i-th
        dipole, :math:`\omega` is the angular frequency, :math:`\mathbf{\mu}_i` is the dipole moment and
        :math:`\mathbf{E}_i` includes the reflections of the dipole field from the layer interfaces, as well as the
        scattered field from all particles and the fields from all other dipoles.
        In contrast to dissipated_power_alternative, this routine uses the particle coupling routines and might be
        faster for many particles and few dipoles.

        Args:
            particle_list (list of smuthi.particles.Particle objects): scattering particles
            layer_system (smuthi.layers.LayerSystem): stratified medium
            k_parallel (ndarray or str): array of in-plane wavenumbers for plane wave expansions. If 'default', use
                                         smuthi.coordinates.default_k_parallel
            azimuthal_angles (ndarray or str): array of azimuthal angles for plane wave expansions. If 'default', use
                                               smuthi.coordinates.default_azimuthal_angles

        Returns:
            dissipated power of each dipole (list of floats)
        """
        
        if self.compute_dissipated_power_by_pwe:
            return self.dissipated_power_alternative(particle_list, 
                                                     layer_system, 
                                                     k_parallel, 
                                                     azimuthal_angles)
        
        sys.stdout.write('Dipole array dissipated power evaluation:\n')
        sys.stdout.flush()

        power_list = []

        x_positions = np.array([dipole.position[0] for dipole in self.dipole_list])
        y_positions = np.array([dipole.position[1] for dipole in self.dipole_list])
        z_positions = np.array([dipole.position[2] for dipole in self.dipole_list])

        e_x_in_list, e_y_in_list, e_z_in_list = [], [], []

        for dipole in tqdm(self.dipole_list, desc='Cross contribution        ', file=sys.stdout,
                           bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
                warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
                warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

                e_x_in, e_y_in, e_z_in = dipole.electric_field(x=x_positions,
                                                               y=y_positions,
                                                               z=z_positions,
                                                               layer_system=layer_system)
            e_x_in_list.append(e_x_in)
            e_y_in_list.append(e_y_in)
            e_z_in_list.append(e_z_in)

        for i, dipole in enumerate(tqdm(self.dipole_list, desc='Self and particle contr.  ', file=sys.stdout,
                                   bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}')):
            e_x_in = sum([e_x_in[i] for i_other, e_x_in in enumerate(e_x_in_list) if i_other != i])
            e_y_in = sum([e_y_in[i] for i_other, e_y_in in enumerate(e_y_in_list) if i_other != i])
            e_z_in = sum([e_z_in[i] for i_other, e_z_in in enumerate(e_z_in_list) if i_other != i])

            p = (dipole.dissipated_power(particle_list, layer_system, False) +
                 dipole.angular_frequency() / 2 * (np.conjugate(dipole.dipole_moment[0]) * e_x_in +
                                                   np.conjugate(dipole.dipole_moment[1]) * e_y_in +
                                                   np.conjugate(dipole.dipole_moment[2]) * e_z_in).imag)
            power_list.append(p)

        return power_list

    def dissipated_power_alternative(self, particle_list, layer_system, k_parallel='default',
                                     azimuthal_angles='default'):
        r"""Compute the power that the dipole collection feeds into the system.

        It is computed according to

        .. math::
            P = \sum_i P_{0, i} + \frac{\omega}{2} \mathrm{Im} (\mathbf{\mu}_i^* \cdot \mathbf{E}_i(\mathbf{r}_i))

        where :math:`P_{0,i}` is the power that the i-th dipole would feed into an infinte homogeneous medium with the
        same refractive index as the layer that contains that dipole, :math:`\mathbf{r}_i` is the location of the i-th
        dipole, :math:`\omega` is the angular frequency, :math:`\mathbf{\mu}_i` is the dipole moment and
        :math:`\mathbf{E}_i` includes the reflections of the dipole field from the layer interfaces, as well as the
        scattered field from all particles and the fields from all other dipoles. In contrast to dissipated_power, this
        routine uses the scattered field piecewise expansion and might be faster for few particles or many dipoles.

        Args:
            particle_list (list of smuthi.particles.Particle objects): scattering particles
            layer_system (smuthi.layers.LayerSystem): stratified medium
            k_parallel (ndarray or str): array of in-plane wavenumbers for plane wave expansions. If 'default', use 
                                         smuthi.coordinates.default_k_parallel
            azimuthal_angles (ndarray or str): array of azimuthal angles for plane wave expansions. If 'default', use 
                                               smuthi.coordinates.default_azimuthal_angles

        Returns:
            dissipated power of each dipole (list of floats)
        """
        sys.stdout.write('Dipole array dissipated power evaluation:\n')
        sys.stdout.flush()
                        
        if k_parallel == 'default':
            k_parallel = coord.default_k_parallel
            
        if azimuthal_angles == 'default':
            azimuthal_angles = coord.default_azimuthal_angles
        
        power_list = []
        
        x_positions = np.array([dipole.position[0] for dipole in self.dipole_list])
        y_positions = np.array([dipole.position[1] for dipole in self.dipole_list])
        z_positions = np.array([dipole.position[2] for dipole in self.dipole_list])
        
        layer_numbers = set([layer_system.layer_number(z) for z in z_positions])
        scat_fld_exp = sf.scattered_field_piecewise_expansion(self.vacuum_wavelength, particle_list, layer_system, 
                                                              k_parallel, azimuthal_angles, layer_numbers)
        
        e_x_scat, e_y_scat, e_z_scat = scat_fld_exp.electric_field(x_positions, y_positions, z_positions)
        
        e_x_in_direct_list, e_y_in_direct_list, e_z_in_direct_list = [], [], []
        e_x_in_response_list, e_y_in_response_list, e_z_in_response_list = [], [], []
        
        for dipole in tqdm(self.dipole_list, desc='Self and cross contrib.   ', file=sys.stdout,
                                        bar_format='{l_bar}{bar}| elapsed: {elapsed} ' 'remaining: {remaining}'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
                warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
                warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
                e_x_in_direct, e_y_in_direct, e_z_in_direct = dipole.electric_field(
                    x=x_positions, y=y_positions, z=z_positions, layer_system=layer_system, include_layer_response=False)
            
            e_x_in_response, e_y_in_response, e_z_in_response = dipole.electric_field(
                x=x_positions, y=y_positions, z=z_positions, layer_system=layer_system, include_direct_field=False)
            
            e_x_in_direct_list.append(e_x_in_direct)
            e_y_in_direct_list.append(e_y_in_direct)
            e_z_in_direct_list.append(e_z_in_direct)
            
            e_x_in_response_list.append(e_x_in_response)
            e_y_in_response_list.append(e_y_in_response)
            e_z_in_response_list.append(e_z_in_response)
            
        for i, dipole in enumerate(self.dipole_list):
            e_x_in = (sum([e_x_in[i] for i_other, e_x_in in enumerate(e_x_in_direct_list) if i_other != i])
                      + sum([e_x_in[i] for i_other, e_x_in in enumerate(e_x_in_response_list)]))
            e_y_in = (sum([e_y_in[i] for i_other, e_y_in in enumerate(e_y_in_direct_list) if i_other != i])
                      + sum([e_y_in[i] for i_other, e_y_in in enumerate(e_y_in_response_list)]))
            e_z_in = (sum([e_z_in[i] for i_other, e_z_in in enumerate(e_z_in_direct_list) if i_other != i])
                      + sum([e_z_in[i] for i_other, e_z_in in enumerate(e_z_in_response_list)]))
            
            p = (dipole.dissipated_power_homogeneous_background(layer_system) +
                 dipole.angular_frequency() / 2 * (np.conjugate(dipole.dipole_moment[0]) * (e_x_scat[i] + e_x_in) +
                                                   np.conjugate(dipole.dipole_moment[1]) * (e_y_scat[i] + e_y_in) +
                                                   np.conjugate(dipole.dipole_moment[2]) * (e_z_scat[i] + e_z_in)).imag)
            power_list.append(p)

        return power_list
    
    @memo.Memoize
    def plane_wave_expansion(self, layer_system, i, k_parallel_array=None, azimuthal_angles_array=None):
        """Plane wave expansion of the dipole collection's field.

        Args:
            layer_system (smuthi.layer.LayerSystem):    stratified medium
            i (int):                                    layer number in which to evaluate the expansion
            k_parallel_array (numpy.ndarray):           in-plane wavenumber array for the expansion
            azimuthal_angles_array (numpy.ndarray):     azimuthal angles for the expansion

        Returns:
            tuple of to smuthi.field_expansion.PlaneWaveExpansion objects, one for upgoing and one for downgoing
            component
        """
        if k_parallel_array is None:
            k_parallel_array = self.k_parallel_array
        if azimuthal_angles_array is None:
            azimuthal_angles_array = self.azimuthal_angles_array        
        
        virtual_particle_list = []

        for dipole in self.dipole_list:
            virtual_particle = part.Particle(position=dipole.position)
            virtual_particle.scattered_field = dipole.outgoing_spherical_wave_expansion(layer_system)
            virtual_particle_list.append(virtual_particle)

        return sf.scattered_field_pwe(self.vacuum_wavelength, virtual_particle_list, layer_system, i, k_parallel_array,
                                      azimuthal_angles_array, include_direct=True, include_layer_response=True)
