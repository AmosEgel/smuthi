"""Functions for the transformation of plane and spherical vector wave 
functions as well as of plane and spherical wave fex."""

import numpy as np
import smuthi.fields.expansions as fex
import smuthi.fields as flds
import smuthi.utility.math as mathfunc
import smuthi.utility.memoizing as memo

###############################################################################
#                          transformations                                    #
###############################################################################


def transformation_coefficients_vwf(tau, l, m, pol, kp=None, kz=None, pilm_list=None, taulm_list=None, dagger=False):
    r"""Transformation coefficients B to expand SVWF in PVWF and vice versa:

    .. math::
        B_{\tau l m,j}(x) = -\frac{1}{\mathrm{i}^{l+1}} \frac{1}{\sqrt{2l(l+1)}} (\mathrm{i} \delta_{j1} + \delta_{j2})
        (\delta_{\tau j} \tau_l^{|m|}(x) + (1-\delta_{\tau j} m \pi_l^{|m|}(x))

    For the definition of the :math:`\tau_l^m` and :math:`\pi_l^m` functions, see
    `A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006
    <https://doi.org/10.1007/978-3-540-33697-6>`_
    
    Compare also section 2.3.3 of [Egel 2018 diss].
    
    Args:
        tau (int):          SVWF polarization, 0 for spherical TE, 1 for spherical TM
        l (int):            l=1,... SVWF multipole degree
        m (int):            m=-l,...,l SVWF multipole order
        pol (int):          PVWF polarization, 0 for TE, 1 for TM
        kp (numpy array):         PVWF in-plane wavenumbers
        kz (numpy array):         complex numpy-array: PVWF out-of-plane wavenumbers
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
        plm_list, pilm_list, taulm_list = mathfunc.legendre_normalized(ct, st, l)

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

    swe = fex.SphericalWaveExpansion(k=pwe.k, l_max=l_max, m_max=m_max, kind='regular', 
                                 reference_point=reference_point, lower_z=pwe.lower_z, 
                                 upper_z=pwe.upper_z)
    kpgrid = pwe.k_parallel_grid()
    agrid = pwe.azimuthal_angle_grid()
    kx = kpgrid * np.cos(agrid)
    ky = kpgrid * np.sin(agrid)
    kz = pwe.k_z_grid()
    kzvec = pwe.k_z()

    kvec = np.array([kx, ky, kz])
    rswe_mn_rpwe = np.array(reference_point) - np.array(pwe.reference_point)

    # phase factor for the translation of the reference point from rvec_iS to rvec_S
    ejkriSS = np.exp(1j * (kvec[0] * rswe_mn_rpwe[0] + kvec[1] * rswe_mn_rpwe[1] + kvec[2] * rswe_mn_rpwe[2]))

    # phase factor times pwe coefficients
    gejkriSS = pwe.coefficients * ejkriSS[None, :, :]  # indices: pol, jk, ja
    
    ct = kzvec / pwe.k
    st = pwe.k_parallel / pwe.k
    plm_list, pilm_list, taulm_list = mathfunc.legendre_normalized(ct, st, l_max)

    for m in range(-m_max, m_max + 1):
        emjma_geijkriSS = np.exp(-1j * m * pwe.azimuthal_angles)[None, None, :] * gejkriSS
        for l in range(max(1, abs(m)), l_max + 1):
            for tau in range(2):
                ak_integrand = np.zeros(kpgrid.shape, dtype=complex)
                for pol in range(2):
                    Bdag = transformation_coefficients_vwf(tau, l, m, pol=pol, pilm_list=pilm_list,
                                                               taulm_list=taulm_list, kz=kzvec, dagger=True)
                    ak_integrand += Bdag[:, None] * emjma_geijkriSS[pol, :, :]
                if len(pwe.k_parallel) > 1:
                    an = np.trapz(np.trapz(ak_integrand, pwe.azimuthal_angles) * pwe.k_parallel, pwe.k_parallel) * 4
                else:
                    an = ak_integrand * 4
                swe.coefficients[flds.multi_to_single_index(tau, l, m, swe.l_max, swe.m_max)] = np.squeeze(an)
    return swe


def swe_to_pwe_conversion(swe, k_parallel, azimuthal_angles, layer_system=None, layer_number=None,
                          layer_system_mediated=False):
    """Convert SphericalWaveExpansion object to a PlaneWaveExpansion object.

    Args:
        swe (SphericalWaveExpansion):             Spherical wave expansion to be converted
        k_parallel (numpy array or str):          In-plane wavenumbers for the pwe object.
        azimuthal_angles (numpy array or str):    Azimuthal angles for the pwe object
        layer_system (smuthi.layers.LayerSystem): Stratified medium in which the origin of the SWE is located
        layer_number (int):                       Layer number in which the PWE should be valid.
        layer_system_mediated (bool):             If True, the PWE refers to the layer system response of the SWE, 
                                                  otherwise it is the direct transform.

    Returns:
        Tuple of two PlaneWaveExpansion objects, first upgoing, second downgoing.
    """
    # todo: manage diverging swe
    k_parallel = np.array(k_parallel, ndmin=1)
    
    i_swe = layer_system.layer_number(swe.reference_point[2])
    if layer_number is None and not layer_system_mediated:
        layer_number = i_swe
    reference_point = [0, 0, layer_system.reference_z(i_swe)]
    lower_z_up = swe.reference_point[2]
    upper_z_up = layer_system.upper_zlimit(layer_number)
    pwe_up = fex.PlaneWaveExpansion(k=swe.k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles, kind='upgoing',
                                    reference_point=reference_point, lower_z=lower_z_up, upper_z=upper_z_up)
    lower_z_down = layer_system.lower_zlimit(layer_number)
    upper_z_down = swe.reference_point[2]
    pwe_down = fex.PlaneWaveExpansion(k=swe.k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles,
                                      kind='downgoing', reference_point=reference_point, lower_z=lower_z_down,
                                      upper_z=upper_z_down)

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
    
    ct_up = pwe_up.k_z() / swe.k
    st_up = pwe_up.k_parallel / swe.k
    plm_list_up, pilm_list_up, taulm_list_up = mathfunc.legendre_normalized(ct_up, st_up, swe.l_max)

    ct_down = pwe_down.k_z() / swe.k
    st_down = pwe_down.k_parallel / swe.k
    plm_list_down, pilm_list_down, taulm_list_down = mathfunc.legendre_normalized(ct_down, st_down, swe.l_max)
    
    for m in range(-swe.m_max, swe.m_max + 1):
        eima = np.exp(1j * m * pwe_up.azimuthal_angles)  # indices: alpha_idx
        for pol in range(2):
            dbB_up = np.zeros(len(k_parallel), dtype=complex)
            dbB_down = np.zeros(len(k_parallel), dtype=complex)
            for l in range(max(1, abs(m)), swe.l_max + 1):
                for tau in range(2):
                    dbB_up += swe.coefficients_tlm(tau, l, m) * transformation_coefficients_vwf(
                        tau, l, m, pol, pilm_list=pilm_list_up, taulm_list=taulm_list_up)
                    dbB_down += swe.coefficients_tlm(tau, l, m) * transformation_coefficients_vwf(
                        tau, l, m, pol, pilm_list=pilm_list_down, taulm_list=taulm_list_down)
            pwe_up.coefficients[pol, :, :] += dbB_up[:, None] * eima[None, :]
            pwe_down.coefficients[pol, :, :] += dbB_down[:, None] * eima[None, :]

    pwe_up.coefficients = pwe_up.coefficients / (2 * np.pi * kzvec[None, :, None] * swe.k) * ejkrSiS_up[None, :, :]
    pwe_down.coefficients = (pwe_down.coefficients / (2 * np.pi * kzvec[None, :, None] * swe.k)
                             * ejkrSiS_down[None, :, :])

    if layer_system_mediated:
        pwe_up, pwe_down = layer_system.response((pwe_up, pwe_down), i_swe, layer_number)

    return pwe_up, pwe_down


###############################################################################
#                          translations                                       #
###############################################################################

def translation_coefficients_svwf(tau1, l1, m1, tau2, l2, m2, k, d, sph_hankel=None, legendre=None, exp_immphi=None):
    r"""Coefficients of the translation operator for the expansion of an outgoing spherical wave in terms of
    regular spherical waves with respect to a different origin:

    .. math::
        \mathbf{\Psi}_{\tau l m}^{(3)}(\mathbf{r} + \mathbf{d} = \sum_{\tau'} \sum_{l'} \sum_{m'}
        A_{\tau l m, \tau' l' m'} (\mathbf{d}) \mathbf{\Psi}_{\tau' l' m'}^{(1)}(\mathbf{r})

    for :math:`|\mathbf{r}|<|\mathbf{d}|`.
    
    See also section 2.3.3 and appendix B of [Egel 2018 diss].
    
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
        sph_hankel = [mathfunc.spherical_hankel(n, k * dd) for n in range(l1 + l2 + 1)]

    if legendre is None:
        costthetd = d[2] / dd
        sinthetd = np.sqrt(d[0] ** 2 + d[1] ** 2) / dd
        legendre, _, _ = mathfunc.legendre_normalized(costthetd, sinthetd, l1 + l2)

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
        sph_bessel = [mathfunc.spherical_bessel(n, k * dd) for n in range(l1 + l2 + 1)]

    if legendre is None:
        costthetd = d[2] / dd
        sinthetd = np.sqrt(d[0] ** 2 + d[1] ** 2) / dd
        legendre, _, _ = mathfunc.legendre_normalized(costthetd, sinthetd, l1 + l2)

    A = complex(0)
    for ld in range(abs(l1 - l2), l1 + l2 + 1):
        a5, b5 = ab5_coefficients(l1, m1, l2, m2, ld)
        if tau1==tau2:
            A += a5 * sph_bessel[ld] * legendre[ld][abs(m1 - m2)]
        else:
            A += b5 * sph_bessel[ld] * legendre[ld][abs(m1 - m2)]
    A = eimph * A
    return A


@memo.Memoize
# @jit(complex128[:](int32, int32, int32, int32, int32),
#      nopython=True, cache=True)
def ab5_coefficients(l1, m1, l2, m2, p):
    """a5 and b5 are the coefficients used in the evaluation of the SVWF translation
    operator.
    
    See section 2.3.3 and appendix B of [Egel 2018 diss].

    Args:
        l1 (int):           l=1,...: Original wave's SVWF multipole degree
        m1 (int):           m=-l,...,l: Original wave's SVWF multipole order
        l2 (int):           l=1,...: Partial wave's SVWF multipole degree
        m2 (int):           m=-l,...,l: Partial wave's SVWF multipole order
        p (int):            p parameter

    Returns:
        A tuple (a5, b5) where a5 and b5 are complex.
    """
    jfac = 1.0j ** (abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * (-1) ** (m1 - m2)
    fac1 = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1)))
    fac2a = (l1 * (l1 + 1) + l2 * (l2 + 1) - p * (p + 1)) * np.sqrt(2 * p + 1)
    fac2b = np.sqrt((l1 + l2 + 1 + p) * (l1 + l2 + 1 - p) * (p + l1 - l2) * (p - l1 + l2) * (2 * p + 1))
    # Note that arguments are in two_j = 2*j.
    wig1 = mathfunc.nb_wig3jj(2*l1, 2*l2, 2*p, 2*m1, -m2*2, -(m1 - m2)*2)
    wig2a = mathfunc.nb_wig3jj(2*l1, 2*l2, 2*p, 0, 0, 0)
    wig2b = mathfunc.nb_wig3jj(2*l1, 2*l2, 2*(p - 1), 0, 0, 0)

    a = jfac * fac1 * fac2a * wig1 * wig2a
    b = jfac * fac1 * fac2b * wig1 * wig2b
    return np.array([a, b])


###############################################################################
#                          rotations                                          #
###############################################################################

def block_rotation_matrix_D_svwf(l_max, m_max, alpha, beta, gamma, wdsympy=False):
    """Rotation matrix for the rotation of SVWFs between the labratory 
    coordinate system (L) and a rotated coordinate system (R)
    
    Args:
        l_max (int):      Maximal multipole degree
        m_max (int):      Maximal multipole order
        alpha (float):    First Euler angle, rotation around z-axis, in rad
        beta (float):     Second Euler angle, rotation around y'-axis in rad
        gamma (float):    Third Euler angle, rotation around z''-axis in rad
        wdsympy (bool):   If True, Wigner-d-functions come from the sympy toolbox
        
    Returns:
        rotation matrix of dimension [blocksize, blocksize]
    """
    
    b_size = flds.blocksize(l_max, m_max)
    rotation_matrix = np.zeros([b_size, b_size], dtype=complex)
    
    for l in range(l_max + 1):
        mstop = min(l, m_max)
        for m1 in range(-mstop, mstop + 1):
            for m2 in range(-mstop, mstop + 1):
                rotation_matrix_coefficient = mathfunc.wigner_D(l, m1, m2, alpha, beta, gamma, wdsympy)
                for tau in range(2):
                    n1 = flds.multi_to_single_index(tau, l, m1, l_max, m_max)
                    n2 = flds.multi_to_single_index(tau, l, m2, l_max, m_max)
                    rotation_matrix[n1, n2] = rotation_matrix_coefficient

    return rotation_matrix
