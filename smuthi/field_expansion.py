import numpy as np


class SphericalWaveExpansion:
    r"""A class to manage spherical wave expansions of the form

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{\tau} \sum_l \sum_m (a_{\tau l m} \Psi^{(1)}_{\tau l m}(\mathbf{r}-\mathbf{r}_0)
        + b_{\tau l m} \Psi^{(3)}_{\tau l m}(\mathbf{r}-\mathbf{r}_0)),

    where :math:`\mathbf{\Psi}_{\tau l m}^{(1,3)}` are the regular and outgoing SVWFs, see
    :meth:`smuthi.vector_wave_functions.spherical_vector_wave_function`, :math:`\mathbf{r}_0` is the reference point of
    the expansion, and :math:`\tau=1,2` and
    :math:`l=1,...,l_\mathrm{max}` whereas :math:`m=-\min(l,m_\mathrm{max}),...,\min(l,m_\mathrm{max})`.

    Args:
        l_max (int):            Truncation multipole degree of spherical wave expansion, :math:`1\leq l_\mathrm{max}`
        m_max (int):            Truncation multipole order of spherical wave expansion,
                                :math:`0\leq m_\mathrm{max} \leq l_\mathrm{max}`
    """

    def __init__(self, l_max=None, m_max=None):
        self.l_max = l_max
        if m_max is None:
            self.m_max = l_max
        else:
            self.m_max = m_max
        self.regular_coefficients = np.zeros(self.number_of_indices(), dtype=complex)  # the a_tlm
        self.outgoing_coefficients = np.zeros(self.number_of_indices(), dtype=complex)  # the b_tlm

    def multi_to_single_index(self, tau, l, m):
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
            tau (int):      polarization index :math:`\tau`(0=spherical TE, 1=spherical TM)
            l (int):              degree :math:`l` (1, ..., lmax)
            m (int):              order :math:`m` (-min(l,mmax),...,min(l,mmax))

        Returns:
            single index (int) subsuming :math:`(\tau, l, m)`
        """
        # use:
        # \sum_{l=1}^lmax (2\min(l,mmax)+1) = \sum_{l=1}^mmax (2l+1) + \sum_{l=mmax+1}^lmax (2mmax+1)
        #                                   = 2*(1/2*mmax*(mmax+1))+mmax  +  (lmax-mmax)*(2*mmax+1)
        #                                   = mmax*(mmax+2)               +  (lmax-mmax)*(2*mmax+1)
        tau_blocksize = self.m_max * (self.m_max + 2) + (self.l_max - self.m_max) * (2 * self.m_max + 1)
        n = tau * tau_blocksize
        if (l - 1) <= self.m_max:
            n += (l - 1) * (l - 1 + 2)
        else:
            n += self.m_max * (self.m_max + 2) + (l - 1 - self.m_max) * (2 * self.m_max + 1)
        n += m + min(l, self.m_max)
        return n

    def number_of_indices(self):
        """
        Returns:
             total number of indices which is the maximal index plus 1."""
        return self.multi_to_single_index(tau=1, l=self.l_max, m=self.m_max) + 1

    def regular_expansion_coefficient(self, tau, l, m):
        r"""Single regular expansion coefficient for fixed SWE index combination

        Args:
            tau (int):      polarization index :math:`\tau`(0=spherical TE, 1=spherical TM)
            l (int):        degree :math:`l` (1, ..., lmax)
            m (int):        order :math:`m` (-min(l,mmax),...,min(l,mmax))

        Returns:
            expansion coefficient :math:`a_{\tau l m}`
        """
        return self.regular_coefficients[self.multi_to_single_index(tau, l, m)]

    def outgoing_expansion_coefficient(self, tau, l, m):
        r"""Single outgoing expansion coefficient for fixed SWE index combination

        Args:
            tau (int):      polarization index :math:`\tau`(0=spherical TE, 1=spherical TM)
            l (int):        degree :math:`l` (1, ..., lmax)
            m (int):        order :math:`m` (-min(l,mmax),...,min(l,mmax))

        Returns:
            outgoing expansion coefficient :math:`b_{\tau l m}`
        """
        return self.outgoing_coefficients[self.multi_to_single_index(tau, l, m)]

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
