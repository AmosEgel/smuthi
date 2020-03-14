"""This subpackage contains functionality that has to do with the
representation of electromagnetic fields in spherical or plane vector wave 
functions.
The __init__ module contains some helper functions (e.g. with respect to
SVWF indexing) and is the place to store default coordinate arrays for
Sommerfeld integrals and field expansions."""

import numpy as np


def angular_frequency(vacuum_wavelength):
    """Angular frequency :math:`\omega = 2\pi c / \lambda`

    Args:
        vacuum_wavelength (float): Vacuum wavelength in length unit

    Returns:
        Angular frequency in the units of c=1 (time units=length units). 
        This is at the same time the vacuum wavenumber.
    """
    return 2 * np.pi / vacuum_wavelength


###############################################################################
#                         SWE indexing                                        #
###############################################################################

def multi_to_single_index(tau, l, m, l_max, m_max):
    r"""Unique single index for the totality of indices characterizing a svwf expansion coefficient.

    The mapping follows the scheme:

    +-------------+-------------+-------------+------------+
    |single index |    spherical wave expansion indices    |
    +=============+=============+=============+============+
    | :math:`n`   | :math:`\tau`| :math:`l`   | :math:`m`  |
    +-------------+-------------+-------------+------------+
    |     1       |     1       |      1      |   -1       |
    +-------------+-------------+-------------+------------+
    |     2       |     1       |      1      |    0       |
    +-------------+-------------+-------------+------------+
    |     3       |     1       |      1      |    1       |
    +-------------+-------------+-------------+------------+
    |     4       |     1       |      2      |   -2       |
    +-------------+-------------+-------------+------------+
    |     5       |     1       |      2      |   -1       |
    +-------------+-------------+-------------+------------+
    |     6       |     1       |      2      |    0       |
    +-------------+-------------+-------------+------------+
    |    ...      |    ...      |     ...     |   ...      |
    +-------------+-------------+-------------+------------+
    |    ...      |     1       |    l_max    |    m_max   |
    +-------------+-------------+-------------+------------+
    |    ...      |     2       |      1      |   -1       |
    +-------------+-------------+-------------+------------+
    |    ...      |    ...      |     ...     |   ...      |
    +-------------+-------------+-------------+------------+

    Args:
        tau (int):      Polarization index :math:`\tau`(0=spherical TE, 1=spherical TM)
        l (int):        Degree :math:`l` (1, ..., lmax)
        m (int):        Order :math:`m` (-min(l,mmax),...,min(l,mmax))
        l_max (int):    Maximal multipole degree
        m_max (int):    Maximal multipole order

    Returns:
        single index (int) subsuming :math:`(\tau, l, m)`
    """
    # use:
    # \sum_{l=1}^lmax (2\min(l,mmax)+1) = \sum_{l=1}^mmax (2l+1) + \sum_{l=mmax+1}^lmax (2mmax+1)
    #                                   = 2*(1/2*mmax*(mmax+1))+mmax  +  (lmax-mmax)*(2*mmax+1)
    #                                   = mmax*(mmax+2)               +  (lmax-mmax)*(2*mmax+1)
    tau_blocksize = m_max * (m_max + 2) + (l_max - m_max) * (2 * m_max + 1)
    n = tau * tau_blocksize
    if (l - 1) <= m_max:
        n += (l - 1) * (l - 1 + 2)
    else:
        n += m_max * (m_max + 2) + (l - 1 - m_max) * (2 * m_max + 1)
    n += m + min(l, m_max)
    return n


def blocksize(l_max, m_max):
    """Number of coefficients in outgoing or regular spherical wave expansion for a single particle.

    Args:
        l_max (int):    Maximal multipole degree
        m_max (int):    Maximal multipole order
    Returns:
         Number of indices for one particle, which is the maximal index plus 1."""
    return multi_to_single_index(tau=1, l=l_max, m=m_max, l_max=l_max, m_max=m_max) + 1


###############################################################################
#                        PWE wave vectors                                     #
###############################################################################

def k_z(k_parallel=None, n_effective=None, k=None, omega=None, vacuum_wavelength=None, refractive_index=None):
    """z-component :math:`k_z=\sqrt{k^2-\kappa^2}` of the wavevector. The branch cut is defined such that the imaginary
    part is not negative, compare section 2.3.1 of [Egel 2018 dissertation].
    Not all of the arguments need to be specified.

    Args:
        k_parallel (numpy ndarray):     In-plane wavenumber :math:`\kappa` (inverse length)
        n_effective (numpy ndarray):    Effective refractive index :math:`n_\mathrm{eff}`
        k (float):                      Wavenumber (inverse length)
        omega (float):                  Angular frequency :math:`\omega` or vacuum wavenumber (inverse length, c=1)
        vacuum_wavelength (float):      Vacuum wavelength :math:`\lambda` (length)
        refractive_index (complex):     Refractive index :math:`n_i` of material

    Returns:
        z-component :math:`k_z` of wavenumber with non-negative imaginary part (inverse length)
    """
    if k_parallel is None:
        if omega is None:
            omega = angular_frequency(vacuum_wavelength)
        k_parallel = n_effective * omega

    if k is None:
        if omega is None:
            omega = angular_frequency(vacuum_wavelength)
        k = refractive_index * omega

    kz = np.sqrt(k ** 2 - k_parallel ** 2 + 0j)
    kz = (kz.imag >= 0) * kz + (kz.imag < 0) * (-kz)  # Branch cut such to prohibit negative imaginary
    return kz


"""The default arrays for k_parrallel, azimuthal_angles and polar_angles are 
used in Sommerfeld integrals or in plane wave expansions whenever no other
arrays for the specification of the wavevectors are explicitly stated."""

# Default arrays for angular coordinates in PlaneWave expansions (azim.) or FarField distributions (polar, azim.)
default_azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180
default_polar_angles = np.arange(0, 181, 1, dtype=float) * np.pi / 180

# Default n_effective array for Sommerfeld integrals - needs to be set, e.g. at beginning of simulation
default_Sommerfeld_k_parallel_array = None

# Default n_effective array for the initial field (beams, dipoles) - needs to be set, e.g. at beginning of simulation
default_initial_field_k_parallel_array = None


def reasonable_neff_waypoints(layer_refractive_indices=None, neff_imag=1e-2, neff_max=None, neff_max_offset=1):
    """Construct a reasonable list of waypoints for a k_parallel array of plane wave expansions.
    The waypoints mark a contour through the complex plane such that possible waveguide mode and
    branchpoint singularity locations are avoided (see section 3.10.2.1 of [Egel 2018 dissertation]).

    Args:
        layer_refractive_indices (list or array):       Complex refractive indices of the plane layer system
        neff_imag (float):                              Extent of the contour into the negative imaginary direction
                                                        (in terms of effective refractive index, n_eff=kappa/omega).
        neff_max (float):                               Truncation value of contour (in terms of effective refractive
                                                        index).
        neff_max_offset (float):                        If no value for `neff_max` is specified, use the last estimated
                                                        singularity location plus this value (in terms of effective
                                                        refractive index). Default=1

    Returns:
        List of complex waypoint values.
    """
    if layer_refractive_indices is None:
        layer_refractive_indices = [1, 2]
    # avoid the real axis on a region between n_min-0.1 and n_max+0.2
    # this should save us from branchpoints and numerically critical waveguide modes
    # (SPPs can be outside of that interval but are then strongly damped)
    min_waveguide_neff = max(0, min(np.array(layer_refractive_indices).real) - 0.1)
    max_waveguide_neff = max(np.array(layer_refractive_indices).real) + 0.2
    if neff_max is None:
        neff_max = max_waveguide_neff + neff_max_offset

    waypoints = [0,
                 min_waveguide_neff,
                 min_waveguide_neff - 1j * neff_imag,
                 max_waveguide_neff - 1j * neff_imag,
                 max_waveguide_neff]
    if neff_max > max_waveguide_neff:
        waypoints.append(neff_max)
    return waypoints


def create_neff_array(neff_waypoints, neff_resolution):
    """Construct an array of complex effective refractive index values. The effective refractive index is a
    dimensionless quantity that will be multiplied by vacuum wavenumber to yield the in-plane component of a wave vector.
    This is used for the plane wave expansion of fields and for Sommerfeld integrals. Complex contours are used to
    improve numerical stability (see section 3.10.2.1 of [Egel 2018 dissertation]).

    Args:
        neff_waypoints (list or ndarray): Corner points through which the contour runs
        neff_resolution(float):           Resolution of contour (i.e., distance between adjacent elements)

    Returns:
        Array of complex effective refractive index values
    """
    neff_segments = []
    for i in range(len(neff_waypoints) - 1):
        abs_dneff = abs(neff_waypoints[i + 1] - neff_waypoints[i])
        if abs_dneff > 0:
            neff_segments.append(
                neff_waypoints[i] + np.arange(0, 1 + neff_resolution / abs_dneff / 2, neff_resolution / abs_dneff,
                                              dtype=complex)
                * (neff_waypoints[i + 1] - neff_waypoints[i]))
    return np.concatenate(neff_segments)


def create_k_parallel_array(vacuum_wavelength, neff_waypoints, neff_resolution):
    """Construct an array of complex in-plane wavenumbers (i.e., the radial component of the cylindrical coordinates of
    the wave-vector). This is used for the plane wave expansion of fields and for Sommerfeld integrals.
    Complex contours are used to improve numerical stability
    (see section 3.10.2.1 of [Egel 2018 dissertation]).

    Args:
        vacuum_wavelength (float):        Vacuum wavelength :math:`\lambda` (length)
        neff_waypoints (list or ndarray): Corner points through which the contour runs
                                          This quantity is dimensionless (effective
                                          refractive index, will be multiplied by vacuum
                                          wavenumber)
        neff_resolution(float):           Resolution of contour, again in terms of
                                          effective refractive index

    Returns:
        Array :math:`\kappa_i` of in-plane wavenumbers (inverse length)
    """
    return create_neff_array(neff_waypoints, neff_resolution) * angular_frequency(vacuum_wavelength)


def branchpoint_correction(layer_refractive_indices, n_effective_array, neff_minimal_branchpoint_distance):
    """Check if an array of complex effective refractive index values (e.g. for Sommerfeld integration) contains
    possible branchpoint singularities and if so, replace them by nearby non-singular locations.

    Args:
        layer_refractive_indices (list or array):       Complex refractive indices of planarly layered medium
        n_effective_array (1d numpy.array):             Complex effective refractive indexc values that are to be checked
                                                        for branchpoint collision
                                                        This array is changed during the function evaluation!
        neff_minimal_branchpoint_distance (float):      Minimal distance that contour points shall have from
                                                        branchpoint singularities
    """
    for n in layer_refractive_indices:
        while True:
            branchpoint_indices = np.where(abs(n_effective_array - n) < neff_minimal_branchpoint_distance)[0]
            if len(branchpoint_indices) == 0:
                break
            idx = branchpoint_indices[0]
            # replace contour point by two points at the middle towards its left and right neighbors
            if not idx == len(n_effective_array) - 1:
                n_effective_array = np.insert(n_effective_array,
                                             idx + 1,
                                             (n_effective_array[idx] + n_effective_array[idx+1]) / 2.0)
                # make sure the new point is ok, otherwise remove
                if abs(n_effective_array[idx + 1] - n) < neff_minimal_branchpoint_distance:
                    n_effective_array = np.delete(n_effective_array, idx + 1)
            if not idx == 0:
                n_effective_array[idx] = (n_effective_array[idx-1] + n_effective_array[idx]) / 2.0
                # make sure the shifted point is ok, otherwise remove
                if abs(n_effective_array[idx] - n) < neff_minimal_branchpoint_distance:
                    n_effective_array = np.delete(n_effective_array, idx)


def reasonable_Sommerfeld_neff_contour(neff_waypoints=None, layer_refractive_indices=None, neff_imag=1e-2,
                                       neff_max=None, neff_max_offset=1, neff_resolution=1e-2,
                                       neff_minimal_branchpoint_distance=None):
    """
    Return a reasonable n_effective array that is suitable for the construction of a Sommerfeld k_parallel integral
    contour. Use this function if you don't want to care for numerical details of your simulation.

    Args:
        neff_waypoints (list or ndarray): Corner points through which the contour runs
                                          This quantity is dimensionless (effective
                                          refractive index, will be multiplied by vacuum
                                          wavenumber)
                                          If not provided, reasonable waypoints are estimated.
        layer_refractive_indices (list):  Complex refractive indices of planarly layered medium
                                          Only needed when no neff_waypoints are provided
        neff_imag (float):                Extent of the contour into the negative imaginary direction
                                          (in terms of effective refractive index, n_eff=kappa/omega).
                                          Only needed when no neff_waypoints are provided
        neff_max (float):                 Truncation value of contour (in terms of effective refractive index).
                                          Only needed when no neff_waypoints are provided
        neff_max_offset (float):          Use the last estimated singularity location plus this value (in terms of
                                          effective refractive index). Default=1
                                          Only needed when no neff_waypoints are provided and if no value for `neff_max`
                                          is specified.
        neff_resolution(float):           Resolution of contour, again in terms of effective refractive index
        neff_minimal_branchpoint_distance (float):      Minimal distance that contour points shall have from
                                                        branchpoint singularities (in terms of effective refractive
                                                        index). This is only relevant if not deflected into imaginary.
                                                        Default: One fifth of neff_resolution

    Returns:
        Array of complex effective refractive index values

    """
    if neff_waypoints is None:
        neff_waypoints = reasonable_neff_waypoints(layer_refractive_indices=layer_refractive_indices,
                                                   neff_imag=neff_imag,
                                                   neff_max=neff_max,
                                                   neff_max_offset=neff_max_offset)
    if neff_minimal_branchpoint_distance is None:
        neff_minimal_branchpoint_distance = neff_resolution / 5
    n_effective = create_neff_array(neff_waypoints, neff_resolution)
    if layer_refractive_indices is not None:
        branchpoint_correction(layer_refractive_indices=layer_refractive_indices,
                               n_effective_array=n_effective,
                               neff_minimal_branchpoint_distance=neff_minimal_branchpoint_distance)
    return n_effective


def reasonable_Sommerfeld_kpar_contour(vacuum_wavelength, neff_waypoints=None, layer_refractive_indices=None,
                                        neff_imag=1e-2, neff_max=None, neff_max_offset=1, neff_resolution=1e-2,
                                        neff_minimal_branchpoint_distance=None):
    """
    Return a reasonable k_parallel array that is suitable as a Sommerfeld integral contour.
    Use this function if you don't want to care for numerical details of your simulation.

    Args:
        vacuum_wavelength (float):        Vacuum wavelength :math:`\lambda` (length)
        neff_waypoints (list or ndarray): Corner points through which the contour runs
                                          This quantity is dimensionless (effective
                                          refractive index, will be multiplied by vacuum
                                          wavenumber)
                                          If not provided, reasonable waypoints are estimated.
        layer_refractive_indices (list):  Complex refractive indices of planarly layered medium
                                          Only needed when no neff_waypoints are provided
        neff_imag (float):                Extent of the contour into the negative imaginary direction
                                          (in terms of effective refractive index, n_eff=kappa/omega).
                                          Only needed when no neff_waypoints are provided
        neff_max (float):                 Truncation value of contour (in terms of effective refractive index).
                                          Only needed when no neff_waypoints are provided
        neff_max_offset (float):          Use the last estimated singularity location plus this value (in terms of
                                          effective refractive index). Default=1
                                          Only needed when no neff_waypoints are provided and if no value for `neff_max`
                                          is specified.
        neff_resolution(float):           Resolution of contour, again in terms of effective refractive index
        neff_minimal_branchpoint_distance (float):      Minimal distance that contour points shall have from
                                                        branchpoint singularities (in terms of effective refractive
                                                        index). This is only relevant if not deflected into imaginary.
                                                        Default: One fifth of neff_resolution

    Returns:
        Array :math:`\kappa_i` of in-plane wavenumbers (inverse length)

    """
    n_effective = reasonable_Sommerfeld_neff_contour(neff_waypoints=neff_waypoints,
                                                     layer_refractive_indices=layer_refractive_indices,
                                                     neff_imag=neff_imag,
                                                     neff_max=neff_max,
                                                     neff_max_offset=neff_max_offset,
                                                     neff_resolution=neff_resolution,
                                                     neff_minimal_branchpoint_distance=neff_minimal_branchpoint_distance)
    return n_effective * angular_frequency(vacuum_wavelength)
