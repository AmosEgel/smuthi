import numpy as np
import smuthi.coordinates as coord
import smuthi.index_conversion as idx
import smuthi.vector_wave_functions as vwf
import smuthi.layers as lay


def plane_wave_pattern_rs(n_effective=None, azimuthal_angles=None, vacuum_wavelength=None,
                          particle_collection=None, linear_system=None, layer_system=None, layer_numbers=None,
                          layerresponse_precision=None):
    """Layer system response plane wave pattern of the scattered field.
    Return a list of plane wave patterns as ndarrays of shape 2 x 2 x nk x na where nk =len(n_effective) and
    na = len(azimuthal_angles).
    The indices are:
    - polarization (0=TE, 1=TM)
    - up/down (0=upwards propagating +kz, 1=downwards propagation -kz)
    - n_effective index
    - azimuthal angle index
    A plane wave pattern is the integrand of a 2-dimensional expansion of the layer system
    response to the scattered field in PVWFs.

    TODO: TEST

    n_effective:            (float) array of n_eff values, i.e. k_parallel / omega
    azimuthal_angles:       (float) array of azimuthal angle values (radian)
    vacuum_wavelength:      (float, length unit)
    particle_collection:    smuthi.particles.ParticleCollection object
    linear_system:          smuthi.linear_system.LinearSystem object
    layer_system:           smuthi.layers.LayerSystem object
    layer_numbers:          list of layer numbers in which the plane wave pattern shall be computed
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    omega = coord.angular_frequency(vacuum_wavelength)
    neff_grid, azimuthal_angle_grid = np.meshgrid(n_effective, azimuthal_angles, indexing='ij')
    kpar_grid = neff_grid * omega

    # read out index specs
    lmax = idx.l_max
    mmax = idx.m_max
    blocksize = idx.number_of_indices()

    iS_list = [layer_system.layer_number(prtcl['position'][2]) for prtcl in particle_collection.particles]
    iS_unique = list(set(iS_list))

    kpar = n_effective * omega
    kiS = [None for x in range(layer_system.number_of_layers())]
    kziS = [None for x in range(layer_system.number_of_layers())]
    ziS = [None for x in range(layer_system.number_of_layers())]
    # transformation coefficients
    B = [None for x in range(layer_system.number_of_layers())]
    # layer response, indices: from layer, to layer
    L = [[None for x1 in range(layer_system.number_of_layers())] for x2 in range(layer_system.number_of_layers())]

    m_vec = np.zeros(blocksize, dtype=int)
    for i in iS_unique:
        kiS[i] = omega * layer_system.refractive_indices[i]
        kziS[i] = coord.k_z(k_parallel=kpar, k=kiS[i])
        ziS[i] = layer_system.reference_z(i)
        # transformation coefficients
        B[i] = np.zeros((2, 2, blocksize, len(n_effective)), dtype=complex)  # indices: pol, plus/minus, n, kpar_idx

        for tau in range(2):
            for m in range(-mmax, mmax + 1):
                for l in range(max(1, abs(m)), lmax + 1):
                    n = idx.multi_to_single_index(tau, l, m)
                    m_vec[n] = m
                    for pol in range(2):
                        B[i][pol, 0, n, :] = vwf.transformation_coefficients_VWF(tau, l, m, pol, kpar, kziS[i])
                        B[i][pol, 1, n, :] = vwf.transformation_coefficients_VWF(tau, l, m, pol, kpar, -kziS[i])

        for i_to in layer_numbers:
            L[i][i_to] = np.zeros((2, 2, 2, len(n_effective)), dtype=complex)  # idcs: pol, pl/mn1, pl/mn2, kpar_idx
            for pol in range(2):
                    L[i][i_to][pol, :, :, :] = lay.layersystem_response_matrix(pol, layer_system.thicknesses,
                                                                               layer_system.refractive_indices, kpar,
                                                                               omega, i, i_to, layerresponse_precision)

    pwp = np.zeros((2, 2, len(n_effective), len(azimuthal_angles)), dtype=complex)  # pol, pl/mn, kp_idx, alpha_idx
    pwp_list = [pwp for x in range(len(layer_numbers))]
    eima = np.exp(1j * np.tensordot(m_vec, azimuthal_angles, axes=0))  # indices: n, alpha_idx

    for iprt, prtcl in enumerate(particle_collection.particles):
        i = layer_system.layer_number(prtcl['position'][2])
        pos = prtcl['position']
        emnikrs_grid = np.exp(-1j * kpar_grid * (pos[0] * np.cos(azimuthal_angle_grid)  # indices: kpar_idx, alpha_idx
                                                 + pos[1] * np.sin(azimuthal_angle_grid)))
        ziSS = pos[2] - ziS[i]
        emnikz = np.exp(-1j * kziS[i] * ziSS)                       # indices: kpar_idx
        eikz = np.exp(1j * kziS[i] * ziSS)                          # indices: kpar_idx

        fac0 = 1 / (2 * np.pi) * emnikrs_grid                        # indices: kpar_idx, alpha_idx
        fac1 = 1 / (kziS[i] * kiS[i])                                # index: kpar_idx
        b = linear_system.scattered_field_coefficients[np.newaxis, iprt, :, np.newaxis]      # index: 1,n, 1
        beB = np.zeros((2, 2, blocksize, len(n_effective)), dtype=complex)  # indices: pol, plus/minus, n, kpar_idx
        beB[:, 0, :, :] = b * (emnikz * B[i][:, 0, :, :])
        beB[:, 1, :, :] = b * (eikz * B[i][:, 1, :, :])                 # idcs: pol, pl/mn, n, kpar_idx
        eimabeB = np.tensordot(beB, eima, axes=[2, 0])               # idcs: pol, pl/mn, kpar_idx, alpha_idx

        for i_ln, i_to in enumerate(layer_numbers):
            for plmn1 in range(2):
                for plmn2 in range(2):
                    pwp_list[i_ln][:, plmn1, :, :] += (fac0[np.newaxis, :, :] *
                                                       (fac1[np.newaxis, :, np.newaxis] *
                                                        (L[i][i_to][:, plmn1, plmn2, :, np.newaxis] *
                                                         eimabeB[:, plmn2, :, :])))

    return pwp_list


def plane_wave_pattern_s(n_effective=None, azimuthal_angles=None, vacuum_wavelength=None, particle_collection=None,
                         linear_system=None, layer_system=None, layer_numbers=None):
    """Plane wave pattern of the direct scattered field.
    Return a list of plane wave patterns as ndarrays of shape 2 x 2 x nk x na where nk =len(n_effective) and
    na = len(azimuthal_angles).
    The indices are:
    - polarization (0=TE, 1=TM)
    - up/down (0=upwards propagating +kz, 1=downwards propagation -kz)
    - n_effective index
    - azimuthal angle index
    A plane wave pattern is the integrand of a 2-dimensional expansion of the direct
    scattered field in PVWFs. In each layer, the plane wave pattern is only valid below all particles (minus-component)
    or above all particles (plus-component)

    TODO: CHECK IF VALID!!


    n_effective:            (float) array of n_eff values, i.e. k_parallel / omega
    azimuthal_angles:       (float) array of azimuthal angle values (radian)
    vacuum_wavelength:      (float, length unit)
    particle_collection:    smuthi.particles.ParticleCollection object
    linear_system:          smuthi.linear_system.LinearSystem object
    layer_system:           smuthi.layers.LayerSystem object
    layer_numbers:          list of layer numbers in which the plane wave pattern shall be computed
    layerresponse_precision:    If None, standard numpy is used for the layer response. If int>0, that many decimal
                                digits are considered in multiple precision. (default=None)
    """
    omega = coord.angular_frequency(vacuum_wavelength)
    neff_grid, azimuthal_angle_grid = np.meshgrid(n_effective, azimuthal_angles, indexing='ij')
    kpar_grid = neff_grid * omega

    # read out index specs
    lmax = idx.l_max
    mmax = idx.m_max
    blocksize = idx.number_of_indices()

    kpar = n_effective * omega

    # plane wave pattern
    pwp = np.zeros((2, 2, len(n_effective), len(azimuthal_angles)), dtype=complex)  # pol, pl/mn, kp_idx, alpha_idx
    pwp_list = [pwp for x in range(len(layer_numbers))]

    m_vec = np.zeros(blocksize, dtype=int)
    for tau in range(2):
        for m in range(-mmax, mmax + 1):
            for l in range(max(1, abs(m)), lmax + 1):
                n = idx.multi_to_single_index(tau, l, m)
                m_vec[n] = m
    eima = np.exp(1j * np.tensordot(m_vec, azimuthal_angles, axes=0))  # indices: n, alpha_idx

    for iidx, i in enumerate(layer_numbers):
        kiS = omega * layer_system.refractive_indices[i]
        kziS = coord.k_z(k_parallel=kpar, k=kiS)
        ziS = layer_system.reference_z(i)
        # transformation coefficients
        B = np.zeros((2, 2, blocksize, len(n_effective)), dtype=complex)  # indices are: pol, plus/minus, n, kpar_idx

        for tau in range(2):
            for m in range(-mmax, mmax + 1):
                for l in range(max(1, abs(m)), lmax + 1):
                    n = idx.multi_to_single_index(tau, l, m)
                    for pol in range(2):
                        B[pol, 0, n, :] = vwf.transformation_coefficients_VWF(tau, l, m, pol, kpar, kziS)
                        B[pol, 1, n, :] = vwf.transformation_coefficients_VWF(tau, l, m, pol, kpar, -kziS)

        for iprt, prtcl in enumerate(particle_collection.particles):
            if layer_system.layer_number(prtcl['position'][2]) == i:
                pos = prtcl['position']
                kx_grid = kpar_grid * np.cos(azimuthal_angle_grid)
                ky_grid = kpar_grid * np.sin(azimuthal_angle_grid)
                emnikplrs_grid = np.exp(-1j * (kx_grid * pos[0] + ky_grid * pos[1]
                                               + kziS[:, np.newaxis] * (pos[2] - ziS)))
                emnikmnrs_grid = np.exp(-1j * (kx_grid * pos[0] + ky_grid * pos[1]
                                               - kziS[:, np.newaxis] * (pos[2] - ziS)))
                eirks = np.concatenate([emnikplrs_grid[np.newaxis, np.newaxis, :, :],
                                        emnikmnrs_grid[np.newaxis, np.newaxis, :, :]], axis=1)  # idcs: kp, al

                fac1 = 1 / (2 * np.pi * kziS * kiS)                          # index: kpar_idx

                b = linear_system.scattered_field_coefficients[iprt, :]     # index: n

                bB = b[np.newaxis, np.newaxis, :, np.newaxis] * B       # idcs: pol, pl/mn, n, kp

                pwp = fac1[np.newaxis, np.newaxis, :, np.newaxis] * eirks * np.tensordot(bB, eima, axes=[2, 0])
                pwp_list[iidx] += pwp

    return pwp_list
