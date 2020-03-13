"""Manage post processing steps to evaluate the electric field inside a sphere"""

import sys
from tqdm import tqdm
import smuthi.fields.expansions as fldex
import smuthi.fields.coordinates_and_contours as coord
import smuthi.fields.transformations as trf
import smuthi.linearsystem.tmatrix.t_matrix as tmt


def internal_field_piecewise_expansion(vacuum_wavelength, particle_list, layer_system, k_parallel='default',
                                       azimuthal_angles='default'):
    """Compute a piecewise field expansion of the internal field of spheres.

    Args:
        vacuum_wavelength (float):                  vacuum wavelength
        particle_list (list):                       list of smuthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem):   stratified medium
        k_parallel (numpy.ndarray or str):          in-plane wavenumbers array.
                                                    if 'default', use smuthi.coordinates.default_k_parallel
        azimuthal_angles (numpy.ndarray or str):    azimuthal angles array
                                                    if 'default', use smuthi.coordinates.default_azimuthal_angles

    Returns:
        internal field as smuthi.field_expansion.PiecewiseFieldExpansion object

    """
    intfld = fldex.PiecewiseFieldExpansion()

    for particle in particle_list:
        if type(particle).__name__ == 'Sphere':
            i_part = layer_system.layer_number(particle.position[2])
            k_medium = coord.angular_frequency(vacuum_wavelength) * layer_system.refractive_indices[i_part]
            k_particle = coord.angular_frequency(vacuum_wavelength) * particle.refractive_index

            internal_field = fldex.SphericalWaveExpansion(
                k_particle,
                l_max=particle.l_max,
                m_max=particle.m_max,
                kind='regular',
                reference_point=particle.position)

            internal_field.validity_conditions.append(particle.is_inside)

            for tau in range(2):
                for l in range(1, particle.l_max+1):
                    for m in range(-l, l+1):
                        n = fldex.multi_to_single_index(tau, l, m, particle.l_max, particle.m_max)
                        b_to_c = (tmt.internal_mie_coefficient(tau, l, k_medium, k_particle, particle.radius)
                                  / tmt.mie_coefficient(tau, l, k_medium, k_particle, particle.radius))
                        internal_field.coefficients[n] = particle.scattered_field.coefficients[n] * b_to_c

            intfld.expansion_list.append(internal_field)

    return intfld
