# -*- coding: utf-8 -*-
import numpy as np
import sympy.physics.wigner
import sympy
import smuthi.coordinates as coord
import smuthi.spherical_functions as sf
import smuthi.vector_wave_functions as vwf


class FieldExpansion:
    """Base class for field expansions."""
    def valid(self, x, y, z):
        """To be overwritten."""
        pass

    def diverging(self, x, y, z):
        """To be overwritten."""

    def electric_field(self, x, y, z):
        """To be overwritten."""
        pass


class PiecewiseFieldExpansion(FieldExpansion):
    """Manage a field that is expanded in different ways for different domains."""
    def __init__(self):
        self.expansion_list = []

    def valid(self, x, y, z):
        vld = np.zeros(x.shape, dtype=bool)
        for fex in self.expansion_list:
            vld = np.logical_or(vld, fex.valid(x, y, z))
        return vld

    def diverging(self, x, y, z):
        dvg = np.zeros(x.shape, dtype=bool)
        for fex in self.expansion_list:
            dvg = np.logical_and(dvg, fex.diverging(x, y, z))
        return dvg

    def electric_field(self, x, y, z):
        ex = np.zeros(x.shape, dtype=complex)
        ey = np.zeros(x.shape, dtype=complex)
        ez = np.zeros(x.shape, dtype=complex)
        for fex in self.expansion_list:
            dex, dey, dez = fex.electric_field(x, y, z)
            ex, ey, ez = ex + dex, ey + dey, ez + dez
        return ex, ey, ez


class SphericalWaveExpansion(FieldExpansion):
    # todo doc
    def __init__(self, k, l_max, m_max=None, kind=None, reference_point=None, lower_z=-np.inf, upper_z=np.inf,
                 inner_r=0, outer_r=np.inf):
        self.k = k
        self.l_max = l_max
        if m_max:
            self.m_max = m_max
        else:
            self.m_max = l_max
        self.coefficients = np.zeros(blocksize(self.l_max, self.m_max), dtype=complex)
        self.kind = kind  # 'regular' or 'outgoing'
        self.reference_point = reference_point
        self.lower_z = lower_z
        self.upper_z = upper_z
        self.inner_r = inner_r
        self.outer_r = outer_r

    def valid(self, x, y, z):
        return np.logical_and(z >= self.lower_z, z < self.upper_z)

    def diverging(self, x, y, z):
        r = np.sqrt((x - self.reference_point[0])**2 + (y - self.reference_point[1])**2
                    + (z - self.reference_point[2])**2)
        if self.kind == 'regular':
            return r >= self.outer_r
        if self.kind == 'outgoing':
            return r <= self.inner_r
        else:
            return None

    def coefficients_tlm(self, tau, l, m):
        """SWE coefficient for given (tau, l, m)

        Args:
            tau (int):  SVWF polarization (0 for spherical TE, 1 for spherical TM)
            l (int):    SVWF degree
            m (int):    SVWF order

        Returns:
            SWE coefficient
        """
        n = multi_to_single_index(tau, l, m, self.l_max, self.m_max)
        return self.coefficients[n]

    def electric_field(self, x, y, z):
        """Evaluate the electric near field corresponding to the SWE.

        Args:
            x (numpy array):    x-coordinates of field points
            y (numpy array):    y-coordinates of field points
            z (numpy array):    z-coordinates of field points

        Returns:
            Tuple (E_x, E_y, E_z) of complex electric field values as numpy arrays of the same shape as x
        """
        xr = x[self.valid(x, y, z)] - self.reference_point[0]
        yr = y[self.valid(x, y, z)] - self.reference_point[1]
        zr = z[self.valid(x, y, z)] - self.reference_point[2]
        ex = np.zeros(x.shape, dtype=complex)
        ey = np.zeros(x.shape, dtype=complex)
        ez = np.zeros(x.shape, dtype=complex)
        for tau in range(2):
            for m in range(-self.m_max, self.m_max + 1):
                for l in range(max(1, abs(m)), self.l_max + 1):
                    b = self.coefficients_tlm(tau, l, m)
                    if self.kind == 'regular':
                        Nx, Ny, Nz = vwf.spherical_vector_wave_function(xr, yr, zr, self.k, 1, tau, l, m)
                    elif self.kind == 'outgoing':
                        Nx, Ny, Nz = vwf.spherical_vector_wave_function(xr, yr, zr, self.k, 3, tau, l, m)
                    ex[self.valid(x, y, z)] += b * Nx
                    ey[self.valid(x, y, z)] += b * Ny
                    ez[self.valid(x, y, z)] += b * Nz
        return ex, ey, ez

    def __add__(self, other):
        if not (self.k == other.k and self.l_max == other.l_max and self.m_max == other.m_max
                and self.kind == other.kind and self.reference_point == other.reference_point):
            raise ValueError('SphericalWaveExpansions are inconsistent.')
        swe_sum = SphericalWaveExpansion(k=self.k, l_max=self.l_max, m_max=self.m_max, kind=self.kind,
                                         reference_point=self.reference_point, inner_r=max(self.inner_r, other.inner_r),
                                         outer_r=min(self.outer_r, other.outer_r),
                                         lower_z=max(self.lower_z, other.lower_z),
                                         upper_z=min(self.upper_z, other.upper_z))
        swe_sum.coefficients = self.coefficients + other.coefficients
        return swe_sum


class PlaneWaveExpansion(FieldExpansion):
    r"""A class to manage plane wave expansions of the form

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{j=1}^2 \iint \mathrm{d}^2\mathbf{k}_\parallel \, g_{j}(\kappa, \alpha)
        \mathbf{\Phi}^\pm_j(\kappa, \alpha; \mathbf{r} - \mathbf{r}_i)

    for :math:`\mathbf{r}` located in a layer defined by :math:`z\in [z_{min}, z_{max}]`
    and :math:`\mathrm{d}^2\mathbf{k}_\parallel = \kappa\,\mathrm{d}\alpha\,\mathrm{d}\kappa` and the double integral
    runs over :math:`\alpha\in[0, 2\pi]` and :math:`\kappa\in[0,\kappa_\mathrm{max}]`. Further,
    :math:`\mathbf{\Phi}^\pm_j` are the PVWFs, see :meth:`plane_vector_wave_function`.

    Internally, the expansion coefficients :math:`g_{ij}^\pm(\kappa, \alpha)` are stored as a list of 4-dimensional
    arrays.
    If the attributes k_parallel and azimuthal_angles have only a single entry, a discrete distribution is
    assumed:

    .. math::
        g_{ij}^-(\kappa, \alpha) \sim \delta^2(\mathbf{k}_\parallel - \mathbf{k}_{\parallel, 0})

    .. todo: update attributes doc

    Args:
        n_effective (ndarray):                      :math:`n_\mathrm{eff} = \kappa / \omega`, can be float or complex
                                                    numpy.array
        azimuthal_angles (ndarray):                 :math:`\alpha`, from 0 to :math:`2\pi`
        layer_system (smuthi.layers.LayerSystem):   Layer system in which the field is expanded

    Attributes:
        k_parallel (array): Array of in-plane wavenumbers
        azimuthal_angles (array): Azimuthal propagation angles of partial plane waves
        layer_system (smuthi.layers.LayerSystem): Layer system object to which the plane wave expansion refers.
        coefficients (list of numpy arrays): coefficients[i][j, pm, k, l] contains
            :math:`g^\pm_{ij}(\kappa_{k}, \alpha_{l})`, where :math:`\pm` is + for pm = 0 and :math:`\pm` is - for
            pm = 1, and the coordinates :math:`\kappa_{k}` and :math:`\alpha_{l}` correspond to n_effective[k] times the
            angular frequency and azimuthal_angles[l], respectively.
    """
    # todo: update doc
    def __init__(self, k, k_parallel=None, azimuthal_angles=None, kind=None, reference_point=None, lower_z=-np.inf,
                 upper_z=np.inf):

        self.k = k
        if hasattr(k_parallel, '__len__'):
            self.k_parallel = np.array(k_parallel)
        else:
            self.k_parallel = np.array([k_parallel])
        if hasattr(azimuthal_angles, '__len__'):
            self.azimuthal_angles = np.array(azimuthal_angles)
        else:
            self.azimuthal_angles = np.array([azimuthal_angles])
        self.kind = kind  # 'upgoing' or 'downgoing'
        self.reference_point = reference_point
        self.lower_z = lower_z
        self.upper_z = upper_z

        # The coefficients :math:`g^\pm_{j}(\kappa,\alpha) are represented as a 3-dimensional numpy.ndarray.
        # The indices are:
        # -  polarization (0=TE, 1=TM)
        # - index of the kappa dimension
        # - index of the alpha dimension
        self.coefficients = np.zeros((2, len(self.k_parallel), len(self.azimuthal_angles)), dtype=complex)

    def valid(self, x, y, z):
        return np.logical_and(z >= self.lower_z, z < self.upper_z)

    def diverging(self, x, y, z):
        return np.zeros(x.shape,dtype=bool)

    def k_parallel_grid(self):
        """Meshgrid of n_effective with respect to azimuthal_angles"""
        kp_grid, _ = np.meshgrid(self.k_parallel, self.azimuthal_angles, indexing='ij')
        return kp_grid

    def azimuthal_angle_grid(self):
        """Meshgrid of azimuthal_angles with respect to n_effective"""
        _, a_grid = np.meshgrid(self.k_parallel, self.azimuthal_angles, indexing='ij')
        return a_grid

    def k_z(self):
        if self.kind == 'upgoing':
            kz = coord.k_z(k_parallel=self.k_parallel, k=self.k)
        elif self.kind == 'downgoing':
            kz = -coord.k_z(k_parallel=self.k_parallel, k=self.k)
        else:
            raise ValueError('pwe kind undefined')
        return kz

    def k_z_grid(self):
        if self.kind == 'upgoing':
            kz = coord.k_z(k_parallel=self.k_parallel_grid(), k=self.k)
        elif self.kind == 'downgoing':
            kz = -coord.k_z(k_parallel=self.k_parallel_grid(), k=self.k)
        else:
            raise ValueError('pwe type undefined')
        return kz

    def __add__(self, other):
        if not (np.isclose(self.k, other.k) and all(np.isclose(self.k_parallel, other.k_parallel))
                and all(np.isclose(self.azimuthal_angles, other.azimuthal_angles)) and self.kind == other.kind
                and self.reference_point == other.reference_point):
            raise ValueError('Plane wave expansion are inconsistent.')
        pwe_sum = PlaneWaveExpansion(k=self.k, k_parallel=self.k_parallel, azimuthal_angles=self.azimuthal_angles,
                                     kind=self.kind, reference_point=self.reference_point,
                                     lower_z=max(self.lower_z, other.lower_z),
                                     upper_z=min(self.upper_z, other.upper_z))
        pwe_sum.coefficients = self.coefficients + other.coefficients
        return pwe_sum

    def electric_field(self, x, y, z):
        """Evaluate electric field.

        Args:
            x (numpy array):    x-coordinates of points where to evaluate the field
            y (numpy array):    y-coordinates of points where to evaluate the field
            z (numpy array):    z-coordinates of points where to evaluate the field

        Returns:
            Tuple (E_x, E_y, E_z) of complex electric fields as numpy arrays with the same shape as x, y, z
        """
        ex = np.zeros(x.shape, dtype=complex)
        ey = np.zeros(x.shape, dtype=complex)
        ez = np.zeros(x.shape, dtype=complex)

        xr = x[self.valid(x, y, z)] - self.reference_point[0]
        yr = y[self.valid(x, y, z)] - self.reference_point[1]
        zr = z[self.valid(x, y, z)] - self.reference_point[2]

        kpgrid = self.k_parallel_grid()
        agrid = self.azimuthal_angle_grid()
        kx = kpgrid * np.cos(agrid)
        ky = kpgrid * np.sin(agrid)
        kz = self.k_z_grid()

        kr = np.zeros((len(xr), len(self.k_parallel), len(self.azimuthal_angles)), dtype=complex)
        kr += np.tensordot(xr, kx, axes=0)
        kr += np.tensordot(yr, ky, axes=0)
        kr += np.tensordot(zr, kz, axes=0)
        eikr = np.exp(1j * kr)

        integrand_x = np.zeros((len(xr), len(self.k_parallel), len(self.azimuthal_angles)), dtype=complex)
        integrand_y = np.zeros((len(xr), len(self.k_parallel), len(self.azimuthal_angles)), dtype=complex)
        integrand_z = np.zeros((len(xr), len(self.k_parallel), len(self.azimuthal_angles)), dtype=complex)

        # pol=0
        integrand_x += (-np.sin(agrid) * self.coefficients[0, :, :])[None, :, :] * eikr
        integrand_y += (np.cos(agrid) * self.coefficients[0, :, :])[None, :, :] * eikr

        # pol=1
        integrand_x += (np.cos(agrid) * kz / self.k * self.coefficients[1, :, :])[None, :, :] * eikr
        integrand_y += (np.sin(agrid) * kz / self.k * self.coefficients[1, :, :])[None, :, :] * eikr
        integrand_z += (-kpgrid / self.k * self.coefficients[1, :, :])[None, :, :] * eikr

        if len(self.k_parallel) > 1:
            ex[self.valid(x, y, z)] = np.trapz(np.trapz(integrand_x, self.azimuthal_angles) * self.k_parallel, self.k_parallel)
            ey[self.valid(x, y, z)] = np.trapz(np.trapz(integrand_y, self.azimuthal_angles) * self.k_parallel, self.k_parallel)
            ez[self.valid(x, y, z)] = np.trapz(np.trapz(integrand_z, self.azimuthal_angles) * self.k_parallel, self.k_parallel)
        else:
            ex[self.valid(x, y, z)] = np.squeeze(integrand_x)
            ey[self.valid(x, y, z)] = np.squeeze(integrand_y)
            ez[self.valid(x, y, z)] = np.squeeze(integrand_z)

        return ex, ey, ez


class FarField:
    """Represent the far field intensity of an electromagnetic field.

    A far field of NaN's is initialized. Default angular resolution is 1 degree.

    Args:
        polar_angles (numpy.ndarray):       Polar angles (default: from 0 to 180 degree in steps of 1 degree)
        azimuthal_angles (numpy.ndarray):   Azimuthal angles (default: from 0 to 360 degree in steps of 1 degree)
        signal_type (str):                         Type of the signal (e.g., intensity for power flux far fields).
    """
    def __init__(self, polar_angles=None, azimuthal_angles=None, signal_type='intensity'):
        if polar_angles is None:
            polar_angles = np.arange(0, 181, 1, dtype=float) * np.pi / 180
        if azimuthal_angles is None:
            azimuthal_angles = np.arange(0, 361, 1, dtype=float) * np.pi / 180
        self.polar_angles = polar_angles
        self.azimuthal_angles = azimuthal_angles

        # The far field signal is represented as a 3-dimensional numpy.ndarray.
        # The indices are:
        # - polarization (0=TE, 1=TM)
        # - index of the polar angle
        # - index of the azimuthal angle
        self.signal = np.zeros((2, len(polar_angles), len(azimuthal_angles)))
        self.signal.fill(np.nan)
        self.signal_type = signal_type

    def azimuthal_integral(self):
        if len(self.azimuthal_angles) > 2:
            return np.trapz(self.signal, self.azimuthal_angles[None, None, :]) * np.sin(self.polar_angles[None, :])
        else:
            return None

    def integral(self):
        if len(self.azimuthal_angles) > 2:
            return np.trapz(self.azimuthal_integral(), self.polar_angles[None, :])
        else:
            return None


def pwe_to_swe_conversion(pwe, l_max, m_max, reference_point):
    """Convert plane wave expansion object to a spherical wave expansion object.

    Args:
        pwe (PlaneWaveExpansion):   Plane wave expansion to be converted
        l_max (int):                Maximal multipole degree of spherical wave expansion
        m_max (int):                Maximal multipole order of spherical wave expansion
        reference_point (list):     Coordinates of reference point in the format [x, y, z]

    Returns:
        SphericalWaveExpansion object.
    """

    if reference_point[2] < pwe.lower_z or reference_point[2] > pwe.upper_z:
        raise ValueError('reference point not inside domain of pwe validity')

    swe = SphericalWaveExpansion(k=pwe.k, l_max=l_max, m_max=m_max, kind='regular', reference_point=reference_point,
                                 lower_z=pwe.lower_z, upper_z=pwe.upper_z)
    kpgrid = pwe.k_parallel_grid()
    agrid = pwe.azimuthal_angle_grid()
    kx = kpgrid * np.cos(agrid)
    ky = kpgrid * np.sin(agrid)
    kz = pwe.k_z_grid()
    kzvec = pwe.k_z()

    kvec = np.array([kx, ky, kz])
    rswe_mn_rpwe = np.array(reference_point) - np.array(pwe.reference_point)

    # phase factor for the translation of the reference point from rvec_iS to rvec_S
    ejkriSS = np.exp(1j * np.tensordot(kvec, rswe_mn_rpwe, axes=([0], [0])))

    # phase factor times pwe coefficients
    gejkriSS = pwe.coefficients * ejkriSS[None, :, :]  # indices: pol, jk, ja

    for tau in range(2):
        for m in range(-m_max, m_max + 1):
            emjma = np.exp(-1j * m * pwe.azimuthal_angles)
            for l in range(max(1, abs(m)), l_max + 1):
                ak_integrand = np.zeros(kpgrid.shape, dtype=complex)
                for pol in range(2):
                    Bdag = vwf.transformation_coefficients_vwf(tau, l, m, pol=pol, kp=pwe.k_parallel, kz=kzvec, 
                                                               dagger=True)
                    ak_integrand += (np.outer(Bdag, emjma) * gejkriSS[pol, :, :])
                if len(pwe.k_parallel) > 1:
                    an = np.trapz(np.trapz(ak_integrand, pwe.azimuthal_angles) * pwe.k_parallel, pwe.k_parallel) * 4
                else:
                    an = ak_integrand * 4
                swe.coefficients[multi_to_single_index(tau, l, m, swe.l_max, swe.m_max)] = np.squeeze(an)
    return swe


def swe_to_pwe_conversion(swe, k_parallel=None, azimuthal_angles=None, layer_system=None, layer_number=None,
                          layer_system_mediated=False):
    """Convert SphericalWaveExpansion object to a PlaneWaveExpansion object.

    Args:
        swe (SphericalWaveExpansion):   Spherical wave expansion to be converted
        k_parallel (numpy array):       In-plane wavenumbers for the pwe object
        azimuthal_angles (numpy array): Azimuthal angles for the pwe object
        layer_system (smuthi.layers.LayerSystem):   Stratified medium in which the origin of the SWE is located
        layer_number (int):             Layer number in which the PWE should be valid.
        layer_system_mediated (bool):   If True, the PWE refers to the layer system response of the SWE, otherwise
                                        it is the direct transform.

    Returns:
        PlaneWaveExpansion object.
    """
    # todo: manage diverging swe
    i_swe = layer_system.layer_number(swe.reference_point[2])
    if layer_number is None and not layer_system_mediated:
        layer_number = i_swe
    reference_point = [0, 0, layer_system.reference_z(i_swe)]
    lower_z_up = swe.reference_point[2]
    upper_z_up = layer_system.upper_zlimit(layer_number)
    pwe_up = PlaneWaveExpansion(k=swe.k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles, kind='upgoing',
                                reference_point=reference_point, lower_z=lower_z_up, upper_z=upper_z_up)
    lower_z_down = layer_system.lower_zlimit(layer_number)
    upper_z_down = swe.reference_point[2]
    pwe_down = PlaneWaveExpansion(k=swe.k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles, kind='downgoing',
                                  reference_point=reference_point, lower_z=lower_z_down, upper_z=upper_z_down)

    agrid = pwe_up.azimuthal_angle_grid()
    kpgrid = pwe_up.k_parallel_grid()

    kx = kpgrid * np.cos(agrid)
    ky = kpgrid * np.sin(agrid)
    kz_up = pwe_up.k_z_grid()
    kz_down = pwe_down.k_z_grid()

    kzvec = pwe_up.k_z()

    kvec_up = np.array([kx, ky, kz_up])
    kvec_down = np.array([kx, ky, kz_down])
    rpwe_mn_rswe = np.array(reference_point) - np.array(swe.reference_point)

    # phase factor for the translation of the reference point from rvec_S to rvec_iS
    ejkrSiS_up = np.exp(1j * np.tensordot(kvec_up, rpwe_mn_rswe, axes=([0], [0])))
    ejkrSiS_down = np.exp(1j * np.tensordot(kvec_down, rpwe_mn_rswe, axes=([0], [0])))

    for m in range(-swe.m_max, swe.m_max + 1):
        eima = np.exp(1j * m * pwe_up.azimuthal_angles)  # indices: alpha_idx
        for l in range(max(1, abs(m)), swe.l_max + 1):
            for tau in range(2):
                for pol in range(2):
                    b = swe.coefficients_tlm(tau, l, m)
                    B_up = vwf.transformation_coefficients_vwf(tau, l, m, pol, pwe_up.k_parallel, pwe_up.k_z())
                    pwe_up.coefficients[pol, :, :] += b * B_up[:, None] * eima[None, :]
                    B_down = vwf.transformation_coefficients_vwf(tau, l, m, pol, pwe_down.k_parallel, pwe_down.k_z())
                    pwe_down.coefficients[pol, :, :] += b * B_down[:, None] * eima[None, :]

    pwe_up.coefficients = pwe_up.coefficients / (2 * np.pi * kzvec[None, :, None] * swe.k) * ejkrSiS_up[None, :, :]
    pwe_down.coefficients = (pwe_down.coefficients / (2 * np.pi * kzvec[None, :, None] * swe.k)
                             * ejkrSiS_down[None, :, :])

    if layer_system_mediated:
        pwe_up, pwe_down = layer_system.response((pwe_up, pwe_down), i_swe, layer_number)

    return pwe_up, pwe_down


def piecewise_to_pwe_conversion():
    pass


def pwe_to_ff_conversion(vacuum_wavelength, plane_wave_expansion):
    """Compute the far field of a plane wave expansion object.

    Args:
        vacuum_wavelength (float):  Vacuum wavelength in length units.
        plane_wave_expansion (smuthi.field_expansion.PlaneWaveExpansion):   Plane wave expansion to convert into far
                                                                            field object.

    Returns:
        A smuthi.field_evaluation.FarField object containing the far field intensity.
    """
    omega = coord.angular_frequency(vacuum_wavelength)
    k = plane_wave_expansion.k
    kp = plane_wave_expansion.k_parallel
    if plane_wave_expansion.kind == 'upgoing':
        polar_angles = np.arcsin(kp / k)
    elif plane_wave_expansion.kind == 'downgoing':
        polar_angles = np.pi - np.arcsin(kp / k)
    else:
        raise ValueError('PWE type not specified')
    if any(polar_angles.imag):
        raise ValueError('complex angles are not allowed')
    azimuthal_angles = plane_wave_expansion.azimuthal_angles
    kkz2 = coord.k_z(k_parallel=kp, k=k) ** 2 * k
    intens = (2 * np.pi ** 2 / omega * kkz2[np.newaxis, :, np.newaxis] 
              * abs(plane_wave_expansion.coefficients) ** 2).real
    srt_idcs = np.argsort(polar_angles)  # reversing order in case of downgoing
    ff = FarField(polar_angles=polar_angles[srt_idcs], azimuthal_angles=azimuthal_angles)
    ff.signal = intens[:, srt_idcs, :]
    return ff


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
