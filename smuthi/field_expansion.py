import numpy as np


class SphericalWaveExpansion:
    r"""A class to manage spherical wave expansions of the form

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{S} \sum_{\tau} \sum_l \sum_m c_{S \tau l m}
        \Psi^{(\nu)}_{\tau l m}(\mathbf{r}-\mathbf{r}_S),

    where :math:`\mathbf{\Psi}_{\tau l m}^{(\nu)}` are the regular (:math:`\nu=1`) or outgoing (:math:`\nu=3`) SVWFs,
    see :meth:`smuthi.vector_wave_functions.spherical_vector_wave_function`, :math:`\mathbf{r}_S` is the reference point
    of particle :math:`S` whereas :math:`\tau=1,2`, :math:`l=1,...,l_\mathrm{max}` and
    :math:`m=-\min(l,m_\mathrm{max}),...,\min(l,m_\mathrm{max})` are the indices of the SVWFs.

    The SphericalWaveExpansion object contains the expansion ceofficients :math:`c_{S \tau l m}`, stored as a ndarray,
    as well as metadata and methods to assign the entries of that array to the particles and multipole indices.

    Args:
        particle_collection (smuthi.particles.ParticleCollection):  Particle collection to which the SWE refers.
    """

    def __init__(self, particle_collection):
        self.particle_collection = particle_collection
        self.blocksizes = [blocksize(particle.l_max, particle.m_max) for particle in self.particle_collection]
        self.number_of_coefficients = sum(self.blocksizes)
        self.coefficients = np.zeros(self.number_of_coefficients, dtype=complex)

    def multi_to_collection_index(self, iS, tau, l, m):
        """Index of a given multipole for a given particle in the coefficients array.

        Args:
            iS (int):   Particle number
            tau (int):  Spherical polarization
            l (int):    Multipole degree
            m (int):    Multipole order

        Returns:
            Collection index (int)
        """
        return sum(self.blocksizes[:iS]) + multi_to_single_index(tau, l, m,
                                                                 self.particle_collection.particles[iS].l_max,
                                                                 self.particle_collection.particles[iS].m_max)

    def collection_index_block(self, iS):
        lmax = self.particle_collection.particles[iS].l_max
        mmax = self.particle_collection.particles[iS].m_max
        return np.arange(self.multi_to_collection_index(iS, 0, 1, -mmax),
                         self.multi_to_collection_index(iS, 1, lmax, mmax) + 1, dtype=int)

    def expansion_coefficient(self, iS, tau, l, m):
        r"""Single expansion coefficient for fixed particle and SWE index combination

        Args:
            iS (int):       Particle number
            tau (int):      Polarization index :math:`\tau`(0=spherical TE, 1=spherical TM)
            l (int):        Degree :math:`l` (1, ..., lmax)
            m (int):        Order :math:`m` (-min(l,mmax),...,min(l,mmax))

        Returns:
            expansion coefficient :math:`c_{S \tau l m}`
        """
        return self.coefficients[self.multi_to_collection_index(iS, tau, l, m)]

    def coefficient_block(self, iS):
        return self.coefficients[self.collection_index_block(iS)]

    def electric_field(self, field_points, reference_point, wavenumber):
        pass  # to do


class PlaneWaveExpansion:
    r"""A class to manage plane wave expansions of the form

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{j=1}^2 \iint \mathrm{d}^2\mathbf{k}_\parallel \,
        (g_j^+(\kappa, \alpha) \mathbf{\Phi}^+_j(\kappa, \alpha; \mathbf{r} - \mathbf{r}_0) +
        g_j^-(\kappa, \alpha) \mathbf{\Phi}^-_j(\kappa, \alpha; \mathbf{r} - \mathbf{r}_0) )

    where :math:`\mathrm{d}^2\mathbf{k}_\parallel = \kappa\,\mathrm{d}\alpha\,\mathrm{d}\kappa` and the double integral
    runs over :math:`\alpha\in[0, 2\pi]` and :math:`\kappa\in[0,\kappa_\mathrm{max}`. Further,
    :math:`\mathbf{\Phi}^\pm_j` are the PVWFs, see :meth:`smuthi.vector_wave_functions.plane_vector_wave_function`.

    Args:
        n_effective (ndarray):      :math:`n_\mathrm{eff} = \kappa / \omega`, can be float or complex ndarray

        azimuthal_angles (ndarray): :math:`\alpha`, from 0 to :math:`2\pi`
    """
    def __init__(self, n_effective=None, azimuthal_angles=None):
        self.n_effective = n_effective
        self.azimuthal_angles = azimuthal_angles

        # the indices of the coefficients array are:
        # -  polarization (0=TE, 1=TM)
        # -  pl/mn: (0=forward propagation, 1=backward propagation)
        # - index of the kappa dimension
        # - index of the alpha dimension
        self.coefficients = np.zeros((2, 2, len(self.n_effective), len(self.azimuthal_angles)), dtype=complex)

    def electric_field(self, field_points, reference_point, wavenumber):
        pass  # to do


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
