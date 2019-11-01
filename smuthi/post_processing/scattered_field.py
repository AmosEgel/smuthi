"""Manage post processing steps to evaluate the scattered electric field"""

import sys
import tqdm
import smuthi.fields.expansions as fldex
import smuthi.fields.coordinates_and_contours as coord
import smuthi.fields.transformations as trf


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
            add_up, add_down = trf.swe_to_pwe_conversion(particle.scattered_field, k_parallel, azimuthal_angles,
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
            pu, pd = trf.swe_to_pwe_conversion(swe=particle.scattered_field, k_parallel=k_parallel,
                                                 azimuthal_angles=azimuthal_angles, layer_system=layer_system)
            pwe_up = pwe_up + pu
            pwe_down = pwe_down + pd

        # layer mediated contribution
        if include_layer_response:
            pu, pd = trf.swe_to_pwe_conversion(swe=particle.scattered_field, k_parallel=k_parallel,
                                                 azimuthal_angles=azimuthal_angles, layer_system=layer_system,
                                                 layer_number=layer_number, layer_system_mediated=True)
            pwe_up = pwe_up + pu
            pwe_down = pwe_down + pd

    return pwe_up, pwe_down
