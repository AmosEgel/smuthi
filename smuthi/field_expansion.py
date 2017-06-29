# -*- coding: utf-8 -*-
import numpy as np
import sympy.physics.wigner
import sympy
import smuthi.layers as lay
import smuthi.coordinates as coord
import smuthi.spherical_functions as sf


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


class SphericalWaveExpansion:
    r"""A class to manage spherical wave expansions of the form

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{S} \sum_{\tau} \sum_l \sum_m c_{S \tau l m}
        \Psi^{(\nu)}_{\tau l m}(\mathbf{r}-\mathbf{r}_S),

    where :math:`\mathbf{\Psi}_{\tau l m}^{(\nu)}` are the regular (:math:`\nu=1`) or outgoing (:math:`\nu=3`) SVWFs,
    see :meth:`spherical_vector_wave_function`, :math:`\mathbf{r}_S` is the reference point
    of particle :math:`S` whereas :math:`\tau=1,2`, :math:`l=1,...,l_\mathrm{max}` and
    :math:`m=-\min(l,m_\mathrm{max}),...,\min(l,m_\mathrm{max})` are the indices of the SVWFs.

    The SphericalWaveExpansion object contains the expansion ceofficients :math:`c_{S \tau l m}`, stored as a ndarray,
    as well as metadata and methods to assign the entries of that array to the particles and multipole indices.

    Args:
        particle_collection (smuthi.particles.ParticleCollection):  Particle collection to which the SWE refers.

    Attributes:
        particle_collection (smuthi.particles.ParticleCollection): Particle collection to which the SWE refers.
        blocksizes (list): Blocksizes of the coefficient vector segments belonging to the individual particles.
        number_of_coefficients (int): Total size of the coefficients vector
        coefficients (array): The actual expansion coefficients as an array. The expansion coefficients of the
                              individual particles are concatenated to a single overall array.
    """

    def __init__(self, particle_collection):
        self.particle_collection = particle_collection
        self.blocksizes = [blocksize(particle.l_max, particle.m_max) for particle in self.particle_collection.particles]
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
        """Numpy array of indices that refer to a segment of the coefficients vector that belong to a certain particle.

        Args:
            iS (int): Particle number

        Returns:
            Numpy array of indices for iS.
        """
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
        """Expansion coefficients belonging to a certain particle.

        Args:
            iS (int): Particle number

        Returns:
            Numpy array of expansion coefficients for particle iS.
        """
        return self.coefficients[self.collection_index_block(iS)]

    def electric_field(self, field_points, reference_point, wavenumber):
        pass  # to do


class PlaneWaveExpansion:
    r"""A class to manage plane wave expansions of the form

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{j=1}^2 \iint \mathrm{d}^2\mathbf{k}_\parallel \,
        (g_{ij}^+(\kappa, \alpha) \mathbf{\Phi}^+_j(\kappa, \alpha; \mathbf{r} - \mathbf{r}_i) +
        g_{ij}^-(\kappa, \alpha) \mathbf{\Phi}^-_j(\kappa, \alpha; \mathbf{r} - \mathbf{r}_i) )

    for :math:`\mathbf{r}` located in the :math:`i`-th layer of a layered medium and
    :math:`\mathrm{d}^2\mathbf{k}_\parallel = \kappa\,\mathrm{d}\alpha\,\mathrm{d}\kappa` and the double integral
    runs over :math:`\alpha\in[0, 2\pi]` and :math:`\kappa\in[0,\kappa_\mathrm{max}]`. Further,
    :math:`\mathbf{\Phi}^\pm_j` are the PVWFs, see :meth:`plane_vector_wave_function`.

    Internally, the expansion coefficients :math:`g_{ij}^\pm(\kappa, \alpha)` are stored as a list of 4-dimensional
    arrays.
    If the attributes n_effective and azimuthal_angles have only a single entry, a discrete distribution is
    assumed:

    .. math::
        g_{ij}^-(\kappa, \alpha) \sim \delta^2(\mathbf{k}_\parallel - \mathbf{k}_{\parallel, 0})

    Args:
        n_effective (ndarray):                      :math:`n_\mathrm{eff} = \kappa / \omega`, can be float or complex
                                                    numpy.array
        azimuthal_angles (ndarray):                 :math:`\alpha`, from 0 to :math:`2\pi`
        layer_system (smuthi.layers.LayerSystem):   Layer system in which the field is expanded

    Attributes:
        n_effective (array): Effective refractive index values of plane waves. Can for example be generated with
            smuthi.coordinates.ComplexContour
        azimuthal_angles (array): Azimuthal propagation angles of partial plane waves
        layer_system (smuthi.layers.LayerSystem): Layer system object to which the plane wave expansion refers.
        coefficients (list of numpy arrays): coefficients[i][j, pm, k, l] contains
            :math:`g^\pm_{ij}(\kappa_{k}, \alpha_{l})`, where :math:`\pm` is + for pm = 0 and :math:`\pm` is - for
            pm = 1, and the coordinates :math:`\kappa_{k}` and :math:`\alpha_{l}` correspond to n_effective[k] times the
            angular frequency and azimuthal_angles[l], respectively.
    """
    def __init__(self, n_effective=None, azimuthal_angles=None, layer_system=None):

        self.n_effective = n_effective
        self.azimuthal_angles = azimuthal_angles
        self.layer_system = layer_system

        # The coefficients :math:`g^\pm_{ij}(\kappa,\alpha) are represented as a list of 4-dimensional numpy.ndarrays.
        # The i-th entry of the list corresponds to the plane wave expansion in layer i.
        # In each of the ndarrays, the indices are:
        # -  polarization (0=TE, 1=TM)
        # -  pl/mn: (0=forward propagation, 1=backward propagation)
        # - index of the kappa dimension
        # - index of the alpha dimension
        self.coefficients = [np.zeros((2, 2, len(n_effective), len(azimuthal_angles)), dtype=complex)
                             for i in range(layer_system.number_of_layers())]

    def n_effective_grid(self):
        """Meshgrid of n_effective with respect to azimuthal_angles"""
        neff_grid, _ = np.meshgrid(self.n_effective, self.azimuthal_angles, indexing='ij')
        return neff_grid

    def azimuthal_angle_grid(self):
        """Meshgrid of azimuthal_angles with respect to n_effective"""
        _, a_grid = np.meshgrid(self.n_effective, self.azimuthal_angles, indexing='ij')
        return a_grid

    def response(self, vacuum_wavelength, excitation_layer_number, layer_numbers='all', precision=None):
        """Construct the plane wave expansion of the layer system response to this plane wave expansion.

        Args:
            vacuum_wavelength (float)
            excitation_layer_number (int):  The coefficients of this layer are interpreted as an excitation PWE
            layer_numbers (list or str):    If 'all', propagate field to all layers. Otherwise, only to the layers the
                                            numbers of which are part of that list
            precision (int):                That many digits are regarded in the layer system response (multi precision)
                                            If None, normal double precision numpy algorithms are used.

        Returns:
            :class:`PlaneWaveExpansion` object containing the layer system response in the layers
            specified in the layer_numbers argument.
        """
        if layer_numbers == 'all':
            layer_numbers = range(self.layer_system.number_of_layers())
        omega = coord.angular_frequency(vacuum_wavelength)
        kpar = self.n_effective * omega
        response_pwe = PlaneWaveExpansion(n_effective=self.n_effective, azimuthal_angles=self.azimuthal_angles,
                                          layer_system=self.layer_system)
        gex = self.coefficients[excitation_layer_number]
        for iL in layer_numbers:
            gij = np.zeros((2, 2, len(self.n_effective), len(self.azimuthal_angles)), dtype=complex)
            l_matrix = lay.evaluate_layerresponse_lookup(self.layer_system.thicknesses,
                                                         self.layer_system.refractive_indices,
                                                         kpar, omega, excitation_layer_number, iL, precision)
            for pol in [0, 1]:
                for ud_excite in [0, 1]:
                    for ud_receive in [0, 1]:
                     gij[pol, ud_receive, :, :] += l_matrix[pol, ud_receive, ud_excite, :] * gex[pol, ud_excite, :, :]
            response_pwe.coefficients[iL] = gij

        return response_pwe

    def __add__(self, other):
        if not (self.n_effective == other.n_effective and self.azimuthal_angles == other.azimuthal_angles
                and self.layer_system == other.layer_system):
            raise ValueError('Plane wave expansion are inconsistent.')
        pwe_sum = PlaneWaveExpansion(n_effective=self.n_effective, azimuthal_angles=self.azimuthal_angles,
                                     layer_system=self.layer_system)
        pwe_sum.coefficients = [self.coefficients[i] + other.coefficients[i] for i in range(len(self.coefficients))]
        return pwe_sum

    def electric_field(self, field_points, reference_point):
        """
        .. todo:: implement
        """
        pass

    def spherical_wave_expansion(self, vacuum_wavelength, particle_collection):
        """Regular spherical wave expansion of the field represented by this plane wave expansion.

        .. todo:: Speed up by recycling the Bdag values

        Args:
            vacuum_wavelength (float)
            particle_collection (smuthi.particles.ParticleCollection):  The particle collection for which the SWE is
                                                                        computed (incoming field)

        Returns:
            SWE coefficients as a :class:`SphericalWaveExpansion` object.
        """
        a = SphericalWaveExpansion(particle_collection)
        angular_frequency = coord.angular_frequency(vacuum_wavelength)
        kpvec = self.n_effective * angular_frequency
        ngrid = self.n_effective_grid()
        agrid = self.azimuthal_angle_grid()
        kx = ngrid * np.cos(agrid) * angular_frequency
        ky = ngrid * np.sin(agrid) * angular_frequency

        for i, particle in enumerate(particle_collection.particles):
            lmax = particle.l_max
            mmax = particle.m_max
            iS = self.layer_system.layer_number(particle.position[2])
            k_iS = self.layer_system.refractive_indices[iS] * angular_frequency
            kz_iS = coord.k_z(k_parallel=self.n_effective_grid() * angular_frequency, k=k_iS)

            kz_iS_vec = coord.k_z(k_parallel=self.n_effective * angular_frequency, k=k_iS)

            kvec_pl_iS = np.array([kx, ky, kz_iS])
            kvec_mn_iS = np.array([kx, ky, -kz_iS])

            rvec_iS = np.array([0, 0, self.layer_system.reference_z(iS)])
            rvec_S = np.array(particle.position)

            # phase factors for the translation of the reference point from rvec_iS to rvec_S
            ejkplriSS = np.exp(1j * np.tensordot(kvec_pl_iS, rvec_S - rvec_iS, axes=([0], [0])))
            ejkmnriSS = np.exp(1j * np.tensordot(kvec_mn_iS, rvec_S - rvec_iS, axes=([0], [0])))

            # phase factors times pwe coefficients
            gejkplriSS = self.coefficients[iS][:, 0, :, :] * ejkplriSS[None, :, :]  # indices: pol, jk, ja
            gejkmnriSS = self.coefficients[iS][:, 1, :, :] * ejkmnriSS[None, :, :]

            # indices: n, pol, pl/mn, jk
            Bdag = np.zeros((blocksize(lmax, mmax), 2, 2, len(self.n_effective)), dtype=complex)
            # indices: n, ja
            emjma = np.zeros((blocksize(lmax, mmax),len(self.azimuthal_angle_grid())), dtype=complex)
            for tau in range(2):
                for m in range(-mmax, mmax + 1):
                    emjma_temp =  np.exp(-1j * m * self.azimuthal_angles)
                    for l in range(max(1, abs(m)), lmax + 1):
                        n = multi_to_single_index(tau, l, m, l_max=lmax, m_max=mmax)
                        an_integrand = np.zeros(ngrid.shape, dtype=complex)
                        emjma[n, :] = emjma_temp
                        for pol in range(2):
                            Bdag[n, pol, 0, :] = transformation_coefficients_VWF(tau, l, m, pol=pol, kp=kpvec,
                                                                                 kz=kz_iS_vec, dagger=True)
                            Bdag[n, pol, 1, :] = transformation_coefficients_VWF(tau, l, m, pol=pol, kp=kpvec,
                                                                                 kz=-kz_iS_vec, dagger=True)
                            an_integrand += (np.outer(Bdag[n, pol, 0, :], emjma[n, :]) * gejkplriSS[pol, :, :]
                                             + np.outer(Bdag[n, pol, 1, :], emjma[n, :]) * gejkmnriSS[pol, :, :])

                        if len(self.n_effective) > 1:
                            an = np.trapz(np.trapz(an_integrand, self.azimuthal_angle_grid()) * self.n_effective,
                                          self.n_effective) * 4 * angular_frequency**2
                        else:
                            an = an_integrand * 4
                        a.coefficients[a.multi_to_collection_index(i, tau, l, m)] = an[0, 0]
        return a


def plane_vector_wave_function(x, y, z, kp, alpha, kz, pol):
    r"""Electric field components of plane wave (PVWF).

    .. math::
        \mathbf{\Phi}_j = \exp ( \mathrm{i} \mathbf{k} \cdot \mathbf{r} ) \hat{ \mathbf{e} }_j

    with :math:`\hat{\mathbf{e}}_0` denoting the unit vector in azimuthal direction ('TE' or 's' polarization),
    and :math:`\hat{\mathbf{e}}_1` denoting the unit vector in polar direction ('TM' or 'p' polarization).

    The input arrays should have one of the following dimensions:

        - x,y,z: (N x 1) matrix
        - kp,alpha,kz: (1 x M) matrix
        - Ex, Ey, Ez: (M x N) matrix

    or

        - x,y,z: (M x N) matrix
        - kp,alpha,kz: scalar
        - Ex, Ey, Ez: (M x N) matrix

    Args:
        x (numpy.ndarray): x-coordinate of position where to test the field (length unit)
        y (numpy.ndarray): y-coordinate of position where to test the field
        z (numpy.ndarray): z-coordinate of position where to test the field
        kp (numpy.ndarray): parallel component of k-vector (inverse length unit)
        alpha (numpy.ndarray): azimthal angle of k-vector (rad)
        kz (numpy.ndarray): z-component of k-vector (inverse length unit)
        pol (int): Polarization (0=TE, 1=TM)

    Returns:
        - x-coordinate of PVWF electric field (numpy.ndarray)
        - y-coordinate of PVWF electric field (numpy.ndarray)
        - z-coordinate of PVWF electric field (numpy.ndarray)
    """
    k = np.sqrt(kp**2 + kz**2)
    kx = kp * np.cos(alpha)
    ky = kp * np.sin(alpha)

    scalar_wave = np.exp(1j * (kx * x + ky * y + kz * z))

    if pol == 0:
        Ex = -np.sin(alpha) * scalar_wave
        Ey = np.cos(alpha) * scalar_wave
        Ez = scalar_wave - scalar_wave
    elif pol == 1:
        Ex = np.cos(alpha) * kz / k * scalar_wave
        Ey = np.sin(alpha) * kz / k * scalar_wave
        Ez = -kp / k * scalar_wave
    else:
        raise ValueError('Polarization must be 0 (TE) or 1 (TM)')

    return Ex, Ey, Ez


def spherical_vector_wave_function(x, y, z, k, nu, tau, l, m):
    """Electric field components of spherical vector wave function (SVWF). The conventions are chosen according to
    `A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006
    <https://doi.org/10.1007/978-3-540-33697-6>`_

    Args:
        x (numpy.ndarray):      x-coordinate of position where to test the field (length unit)
        y (numpy.ndarray):      y-coordinate of position where to test the field
        z (numpy.ndarray):      z-coordinate of position where to test the field
        k (float or complex):   wavenumber (inverse length unit)
        nu (int):               1 for regular waves, 3 for outgoing waves
        tau (int):              spherical polarization, 0 for spherical TE and 1 for spherical TM
        l (int):                l=1,... multipole degree (polar quantum number)
        m (int):                m=-l,...,l multipole order (azimuthal quantum number)

    Returns:
        - x-coordinate of SVWF electric field (numpy.ndarray)
        - y-coordinate of SVWF electric field (numpy.ndarray)
        - z-coordinate of SVWF electric field (numpy.ndarray)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    # unit vector in r-direction
    er_x = x / r
    er_y = y / r
    er_z = z / r

    # unit vector in theta-direction
    eth_x = np.cos(theta) * np.cos(phi)
    eth_y = np.cos(theta) * np.sin(phi)
    eth_z = -np.sin(theta)

    # unit vector in phi-direction
    eph_x = -np.sin(phi)
    eph_y = np.cos(phi)
    eph_z = x - x

    cos_thet = np.cos(theta)
    sin_thet = np.sin(theta)
    plm_list, pilm_list, taulm_list = sf.legendre_normalized(cos_thet, sin_thet, l)
    plm = plm_list[l][abs(m)]
    pilm = pilm_list[l][abs(m)]
    taulm = taulm_list[l][abs(m)]

    kr = k * r
    if nu == 1:
        bes = sf.spherical_bessel(l, kr)
        dxxz = sf.dx_xj(l, kr)
    elif nu == 3:
        bes = sf.spherical_hankel(l, kr)
        dxxz = sf.dx_xh(l, kr)
    else:
        raise ValueError('nu must be 1 (regular SVWF) or 3 (outgoing SVWF)')

    eimphi = np.exp(1j * m * phi)
    prefac = 1/np.sqrt(2 * l * (l + 1))
    if tau == 0:
        Ex = prefac * bes * (1j * m * pilm * eth_x - taulm * eph_x) * eimphi
        Ey = prefac * bes * (1j * m * pilm * eth_y - taulm * eph_y) * eimphi
        Ez = prefac * bes * (1j * m * pilm * eth_z - taulm * eph_z) * eimphi
    elif tau == 1:
        Ex = prefac * (l * (l + 1) * bes / kr * plm * er_x +
                       dxxz / kr * (taulm * eth_x + 1j * m * pilm * eph_x)) * eimphi
        Ey = prefac * (l * (l + 1) * bes / kr * plm * er_y +
                       dxxz / kr * (taulm * eth_y + 1j * m * pilm * eph_y)) * eimphi
        Ez = prefac * (l * (l + 1) * bes / kr * plm * er_z +
                       dxxz / kr * (taulm * eth_z + 1j * m * pilm * eph_z)) * eimphi
    else:
        raise ValueError('tau must be 0 (spherical TE) or 1 (spherical TM)')

    return Ex, Ey, Ez


def transformation_coefficients_VWF(tau, l, m, pol, kp=None, kz=None, pilm_list=None, taulm_list=None, dagger=False):
    r"""Transformation coefficients B to expand SVWF in PVWF and vice versa:

    .. math::
        B_{\tau l m,j}(x) = -\frac{1}{\mathrm{i}^{l+1}} \frac{1}{\sqrt{2l(l+1)}} (\mathrm{i} \delta_{j1} + \delta_{j2})
        (\delta_{\tau j} \tau_l^{|m|}(x) + (1-\delta_{\tau j} m \pi_l^{|m|}(x))

    For the definition of the :math:`\tau_l^m` and :math:`\pi_l^m` functions, see
    `A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006
    <https://doi.org/10.1007/978-3-540-33697-6>`_

    Args:
        tau (int):          SVWF polarization, 0 for spherical TE, 1 for spherical TM
        l (int):            l=1,... SVWF multipole degree
        m (int):            m=-l,...,l SVWF multipole order
        pol (int):          PVWF polarization, 0 for TE, 1 for TM
        kp (array):         PVWF in-plane wavenumbers
        kz (array):         complex numpy-array: PVWF out-of-plane wavenumbers
        pilm_list (list):   2D list numpy-arrays: alternatively to kp and kz, pilm and taulm as generated with
                            legendre_normalized can directly be handed
        taulm_list (list):  2D list numpy-arrays: alternatively to kp and kz, pilm and taulm as generated with
                            legendre_normalized can directly be handed
        dagger (bool):      switch on when expanding PVWF in SVWF and off when expanding SVWF in PVWF

    Returns:
        Transformation coefficient as array (size like kp).
    """
    if pilm_list is None:
        k = np.sqrt(kp**2 + kz**2)
        ct = kz / k
        st = kp / k
        plm_list, pilm_list, taulm_list = sf.legendre_normalized(ct, st, l)

    if tau == pol:
        sphfun = taulm_list[l][abs(m)]
    else:
        sphfun = m * pilm_list[l][abs(m)]

    if dagger:
        if pol == 0:
            prefac = -1 / (-1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * (-1j)
        elif pol == 1:
            prefac = -1 / (-1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * 1
        else:
            raise ValueError('pol must be 0 (TE) or 1 (TM)')
    else:
        if pol == 0:
            prefac = -1 / (1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * (1j)
        elif pol ==1:
            prefac = -1 / (1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * 1
        else:
            raise ValueError('pol must be 0 (TE) or 1 (TM)')

    B = prefac * sphfun
    return B


def translation_coefficients_svwf(tau1, l1, m1, tau2, l2, m2, k, d, sph_hankel=None, legendre=None, exp_immphi=None):
    r"""Coefficients of the translation operator for the expansion of an outgoing spherical wave in terms of
    regular spherical waves with respect to a different origin:

    .. math::
        \mathbf{\Psi}_{\tau l m}^{(3)}(\mathbf{r} + \mathbf{d} = \sum_{\tau'} \sum_{l'} \sum_{m'}
        A_{\tau l m, \tau' l' m'} (\mathbf{d}) \mathbf{\Psi}_{\tau' l' m'}^{(1)}(\mathbf{r})

    for :math:`|\mathbf{r}|<|\mathbf{d}|`.

    Args:
        tau1 (int):             tau1=0,1: Original wave's spherical polarization
        l1 (int):               l=1,...: Original wave's SVWF multipole degree
        m1 (int):               m=-l,...,l: Original wave's SVWF multipole order
        tau2 (int):             tau2=0,1: Partial wave's spherical polarization
        l2 (int):               l=1,...: Partial wave's SVWF multipole degree
        m2 (int):               m=-l,...,l: Partial wave's SVWF multipole order
        k (float or complex):   wavenumber (inverse length unit)
        d (list):               translation vectors in format [dx, dy, dz] (length unit)
                                dx, dy, dz can be scalars or ndarrays
        sph_hankel (list):      Optional. sph_hankel[i] contains the spherical hankel funciton of degree i, evaluated at
                                k*d where d is the norm of the distance vector(s)
        legendre (list):        Optional. legendre[l][m] contains the legendre function of order l and degree m,
                                evaluated at cos(theta) where theta is the polar angle(s) of the distance vector(s)

    Returns:
        translation coefficient A (complex)

    """
    # spherical coordinates of d:
    dd = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

    if exp_immphi is None:
        phid = np.arctan2(d[1], d[0])
        eimph = np.exp(1j * (m1 - m2) * phid)
    else:
        eimph = exp_immphi[m1][m2]

    if sph_hankel is None:
        sph_hankel = [sf.spherical_hankel(n, k * dd) for n in range(l1 + l2 + 1)]

    if legendre is None:
        costthetd = d[2] / dd
        sinthetd = np.sqrt(d[0] ** 2 + d[1] ** 2) / dd
        legendre, _, _ = sf.legendre_normalized(costthetd, sinthetd, l1 + l2)

    A = complex(0)
    for ld in range(abs(l1 - l2), l1 + l2 + 1):
        a5, b5 = ab5_coefficients(l1, m1, l2, m2, ld)
        if tau1 == tau2:
            A += a5 * sph_hankel[ld] * legendre[ld][abs(m1 - m2)]
        else:
            A += b5 * sph_hankel[ld] * legendre[ld][abs(m1 - m2)]
    A = eimph * A
    return A


def translation_coefficients_svwf_out_to_out(tau1, l1, m1, tau2, l2, m2, k, d, sph_bessel=None, legendre=None,
                                             exp_immphi=None):
    r"""Coefficients of the translation operator for the expansion of an outgoing spherical wave in terms of
    outgoing spherical waves with respect to a different origin:

    .. math::
        \mathbf{\Psi}_{\tau l m}^{(3)}(\mathbf{r} + \mathbf{d} = \sum_{\tau'} \sum_{l'} \sum_{m'}
        A_{\tau l m, \tau' l' m'} (\mathbf{d}) \mathbf{\Psi}_{\tau' l' m'}^{(3)}(\mathbf{r})

    for :math:`|\mathbf{r}|>|\mathbf{d}|`.

    Args:
        tau1 (int):             tau1=0,1: Original wave's spherical polarization
        l1 (int):               l=1,...: Original wave's SVWF multipole degree
        m1 (int):               m=-l,...,l: Original wave's SVWF multipole order
        tau2 (int):             tau2=0,1: Partial wave's spherical polarization
        l2 (int):               l=1,...: Partial wave's SVWF multipole degree
        m2 (int):               m=-l,...,l: Partial wave's SVWF multipole order
        k (float or complex):   wavenumber (inverse length unit)
        d (list):               translation vectors in format [dx, dy, dz] (length unit)
                                dx, dy, dz can be scalars or ndarrays
        sph_bessel (list):      Optional. sph_bessel[i] contains the spherical Bessel funciton of degree i, evaluated at
                                k*d where d is the norm of the distance vector(s)
        legendre (list):        Optional. legendre[l][m] contains the legendre function of order l and degree m,
                                evaluated at cos(theta) where theta is the polar angle(s) of the distance vector(s)

    Returns:
        translation coefficient A (complex)
    """
    # spherical coordinates of d:
    dd = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

    if exp_immphi is None:
        phid = np.arctan2(d[1], d[0])
        eimph = np.exp(1j * (m1 - m2) * phid)
    else:
        eimph = exp_immphi[m1][m2]

    if sph_bessel is None:
        sph_bessel = [sf.spherical_bessel(n, k * dd) for n in range(l1 + l2 + 1)]

    if legendre is None:
        costthetd = d[2] / dd
        sinthetd = np.sqrt(d[0] ** 2 + d[1] ** 2) / dd
        legendre, _, _ = sf.legendre_normalized(costthetd, sinthetd, l1 + l2)

    A = complex(0), complex(0)
    for ld in range(abs(l1 - l2), l1 + l2 + 1):
        a5, b5 = ab5_coefficients(l1, m1, l2, m2, ld)
        if tau1==tau2:
            A += a5 * sph_bessel[ld] * legendre[ld][abs(m1 - m2)]
        else:
            A += b5 * sph_bessel[ld] * legendre[ld][abs(m1 - m2)]
    A = eimph * A
    return A


def ab5_coefficients(l1, m1, l2, m2, p, symbolic=False):
    """a5 and b5 are the coefficients used in the evaluation of the SVWF translation
    operator. Their computation is based on the sympy.physics.wigner package and is performed with symbolic numbers.

    Args:
        l1 (int):           l=1,...: Original wave's SVWF multipole degree
        m1 (int):           m=-l,...,l: Original wave's SVWF multipole order
        l2 (int):           l=1,...: Partial wave's SVWF multipole degree
        m2 (int):           m=-l,...,l: Partial wave's SVWF multipole order
        p (int):            p parameter
        symbolic (bool):    If True, symbolic numbers are returned. Otherwise, complex.

    Returns:
        A tuple (a5, b5) where a5 and b5 are symbolic or complex.
    """
    jfac = sympy.I ** (abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * (-1) ** (m1 - m2)
    fac1 = sympy.sqrt((2 * l1 + 1) * (2 * l2 + 1) / sympy.S(2 * l1 * (l1 + 1) * l2 * (l2 + 1)))
    fac2a = (l1 * (l1 + 1) + l2 * (l2 + 1) - p * (p + 1)) * sympy.sqrt(2 * p + 1)
    fac2b = sympy.sqrt((l1 + l2 + 1 + p) * (l1 + l2 + 1 - p) * (p + l1 - l2) * (p - l1 + l2) * (2 * p + 1))
    wig1 = sympy.physics.wigner.wigner_3j(l1, l2, p, m1, -m2, -(m1 - m2))
    wig2a = sympy.physics.wigner.wigner_3j(l1, l2, p, 0, 0, 0)
    wig2b = sympy.physics.wigner.wigner_3j(l1, l2, p - 1, 0, 0, 0)

    if symbolic:
        a = jfac * fac1 * fac2a * wig1 * wig2a
        b = jfac * fac1 * fac2b * wig1 * wig2b
    else:
        a = complex(jfac * fac1 * fac2a * wig1 * wig2a)
        b = complex(jfac * fac1 * fac2b * wig1 * wig2b)
    return a, b
