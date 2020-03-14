# -*- coding: utf-8 -*-
"""Classes to manage the expansion of the electric field in plane wave and 
spherical wave basis sets."""

import numpy as np
import smuthi.fields
import smuthi.fields.vector_wave_functions as vwf
import smuthi.fields.expansions_cuda as cu_src
import smuthi.utility.cuda as cu
import copy
import math


class FieldExpansion:
    """Base class for field expansions."""

    def __init__(self):
        self.validity_conditions = []

    def valid(self, x, y, z):
        """Test if points are in definition range of the expansion. 
        Abstract method to be overwritten in child classes.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            definition domain.
        """
        ret = np.ones(x.shape, dtype=bool)
        for check in self.validity_conditions:
            ret = np.logical_and(ret, check(x, y, z))
        return ret

    def diverging(self, x, y, z):
        """Test if points are in domain where expansion could diverge. Virtual 
        method to be overwritten in child 
        classes.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            divergence domain.
        """
        pass

    def electric_field(self, x, y, z):
        """Evaluate electric field. Virtual method to be overwritten in child 
        classes.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            Tuple of (E_x, E_y, E_z) numpy.ndarray objects with the Cartesian 
            coordinates of complex electric field.
        """
        pass


class PiecewiseFieldExpansion(FieldExpansion):
    r"""Manage a field that is expanded in different ways for different 
    domains, i.e., an expansion of the kind
    
    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{i} \mathbf{E}_i(\mathbf{r}),
    
    where
    
    .. math::
        \mathbf{E}_i(\mathbf{r}) = \begin{cases} \tilde{\mathbf{E}}_i(\mathbf{r}) & \text{ if }\mathbf{r}\in D_i \\ 0 & \text{ else} \end{cases}
    
    and :math:`\tilde{\mathbf{E_i}}(\mathbf{r})` is either a plane wave 
    expansion or a spherical wave expansion, and 
    :math:`D_i` is its domain of validity.
    """
    def __init__(self):
        FieldExpansion.__init__(self)
        self.expansion_list = []

    def valid(self, x, y, z):
        """Test if points are in definition range of the expansion.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            definition domain.
        """
        vld = np.zeros(x.shape, dtype=bool)
        for fex in self.expansion_list:
            vld = np.logical_or(vld, fex.valid(x, y, z))

        vld = np.logical_and(vld, FieldExpansion.valid(self, x, y, z))

        return vld

    def diverging(self, x, y, z):
        """Test if points are in domain where expansion could diverge.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            divergence domain.
        """
        dvg = np.zeros(x.shape, dtype=bool)
        for fex in self.expansion_list:
            dvg = np.logical_and(dvg, fex.diverging(x, y, z))
        return dvg
    
    def electric_field(self, x, y, z):
        """Evaluate electric field.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            Tuple of (E_x, E_y, E_z) numpy.ndarray objects with the Cartesian 
            coordinates of complex electric field.
        """
        x, y, z = np.array(x, ndmin=1), np.array(y, ndmin=1), np.array(z, ndmin=1)
        ex = np.zeros(x.shape, dtype=complex)
        ey = np.zeros(x.shape, dtype=complex)
        ez = np.zeros(x.shape, dtype=complex)
        vld = self.valid(x, y, z)
        for fex in self.expansion_list:
            dex, dey, dez = fex.electric_field(x, y, z)
            ex[vld], ey[vld], ez[vld] = ex[vld] + dex[vld], ey[vld] + dey[vld], ez[vld] + dez[vld]
        return ex, ey, ez

    def compatible(self, other):
        """Returns always true, because any field expansion can be added to a 
        piecewise field expansion."""
        return True

    def __add__(self, other):
        """Addition of expansion objects.

        Args:
            other (FieldExpansion):  expansion object to add to this object

        Returns:
            PiecewiseFieldExpansion object as the sum of this expansion and the 
            other
        """
        # todo: testing
        pfe_sum = PiecewiseFieldExpansion()

        if type(other).__name__ == "PiecewiseFieldExpansion":
            added = [False for other_fex in other.expansion_list]
            for self_fex in self.expansion_list:
                fex = copy.deepcopy(self_fex)
                for i, other_fex in enumerate(other.expansion_list):
                    if (not added[i]) and self_fex.compatible(other_fex):
                        fex = fex + other_fex
                        added[i] = True
                pfe_sum.expansion_list.append(fex)
            for i, other_fex in enumerate(other.expansion_list):
                if not added[i]:
                    pfe_sum.expansion_list.append(other_fex)
        else:
            added = False
            for self_fex in self.expansion_list:
                fex = copy.deepcopy(self_fex)
                if (not added) and fex.compatible(other):
                    pfe_sum.expansion_list.append(fex + other)
                    added = True
                else:
                    pfe_sum.expansion_list.append(fex)
            if not added:
                pfe_sum.expansion_list.append(other)

        return pfe_sum


class SphericalWaveExpansion(FieldExpansion):
    r"""A class to manage spherical wave expansions of the form

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{\tau=1}^2 \sum_{l=1}^\infty \sum_{m=-l}^l a_{\tau l m} 
        \mathbf{\Psi}^{(\nu)}_{\tau l m}(\mathbf{r} - \mathbf{r}_i)

    for :math:`\mathbf{r}` located in a layer defined by 
    :math:`z\in [z_{min}, z_{max}]`
    and where :math:`\mathbf{\Psi}^{(\nu)}_{\tau l m}` are the SVWFs, see 
    :meth:`smuthi.vector_wave_functions.spherical_vector_wave_function`.

    Internally, the expansion coefficients :math:`a_{\tau l m}` are stored as a 
    1-dimensional array running over a multi index :math:`n` subsumming over 
    the SVWF indices :math:`(\tau,l,m)`. The 
    mapping from the SVWF indices to the multi
    index is organized by the function :meth:`multi_to_single_index`.
    
    Args:
        k (float):    wavenumber in layer where expansion is valid
        l_max (int):  maximal multipole degree :math:`l_\mathrm{max}\geq 1` 
        where to truncate the expansion. m_max (int):  maximal multipole order 
        :math:`0 \leq m_\mathrm{max} \leq l_\mathrm{max}` where to truncate the 
        expansion.
        kind (str):   'regular' for :math:`\nu=1` or 'outgoing' for :math:`\nu=3`
        reference_point (list or tuple):  [x, y, z]-coordinates of point relative 
                                          to which the spherical waves are 
                                          considered (e.g., particle center).
        lower_z (float):   the expansion is valid on and above that z-coordinate
        upper_z (float):   the expansion is valid below that z-coordinate
        inner_r (float):   radius inside which the expansion diverges 
                           (e.g. circumscribing sphere of particle)
        outer_r (float):   radius outside which the expansion diverges

    Attributes:
        coefficients (numpy ndarray): expansion coefficients 
        :math:`a_{\tau l m}` ordered by multi index n
    """

    def __init__(self, k, l_max, m_max=None, kind=None, reference_point=None, lower_z=-np.inf, upper_z=np.inf,
                 inner_r=0, outer_r=np.inf):
        FieldExpansion.__init__(self)
        self.k = k
        self.l_max = l_max
        if m_max is not None:
            self.m_max = m_max
        else:
            self.m_max = l_max
        self.coefficients = np.zeros(smuthi.fields.blocksize(self.l_max, self.m_max), dtype=complex)
        self.kind = kind  # 'regular' or 'outgoing'
        self.reference_point = reference_point
        self.lower_z = lower_z
        self.upper_z = upper_z
        self.inner_r = inner_r
        self.outer_r = outer_r

    def valid(self, x, y, z):
        """Test if points are in definition range of the expansion.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            definition domain.
        """
        vld = np.logical_and(z >= self.lower_z, z < self.upper_z)
        return np.logical_and(vld, FieldExpansion.valid(self, x, y, z))

    def diverging(self, x, y, z):
        """Test if points are in domain where expansion could diverge.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            divergence domain.
        """
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
        n = smuthi.fields.multi_to_single_index(tau, l, m, self.l_max, self.m_max)
        return self.coefficients[n]
    
    def electric_field(self, x, y, z):
        """Evaluate electric field.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            Tuple of (E_x, E_y, E_z) numpy.ndarray objects with the Cartesian 
            coordinates of complex electric field.
        """
        x = np.array(x, ndmin=1)
        y = np.array(y, ndmin=1)
        z = np.array(z, ndmin=1)

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

    def compatible(self, other):
        """Check if two spherical wave expansions are compatible in the sense 
        that they can be added coefficient-wise

        Args:
            other (FieldExpansion):  expansion object to add to this object

        Returns:
            bool (true if compatible, false else)
        """
        return (type(other).__name__ == "SphericalWaveExpansion" 
                and self.k == other.k 
                and self.l_max == other.l_max
                and self.m_max == other.m_max 
                and self.kind == other.kind
                and self.reference_point == other.reference_point)

    def __add__(self, other):
        """Addition of expansion objects (by coefficients).
        
        Args:
            other (SphericalWaveExpansion):  expansion object to add to this object
        
        Returns:
            SphericalWaveExpansion object as the sum of this expansion and the other
        """
        # todo: allow different l_max
        if not self.compatible(other):
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
    and :math:`\mathrm{d}^2\mathbf{k}_\parallel = \kappa\,\mathrm{d}\alpha\,\mathrm{d}\kappa`. 
    
    The double integral runs over :math:`\alpha\in[0, 2\pi]` and :math:`\kappa\in[0,\kappa_\mathrm{max}]`. 
    Further, :math:`\mathbf{\Phi}^\pm_j` are the PVWFs, see :meth:`plane_vector_wave_function`.

    Internally, the expansion coefficients :math:`g_{ij}^\pm(\kappa, \alpha)` 
    are stored as a 3-dimensional numpy ndarray.
    
    If the attributes k_parallel and azimuthal_angles have only a single entry, 
    a discrete distribution is assumed:

    .. math::
        g_{j}^\pm(\kappa, \alpha) \sim \delta^2(\mathbf{k}_\parallel - \mathbf{k}_{\parallel, 0})

    .. todo: update attributes doc

    Args:
        k (float):                          wavenumber in layer where expansion is valid
        k_parallel (numpy ndarray):         array of in-plane wavenumbers (can be float or complex)
        azimuthal_angles (numpy ndarray):   :math:`\alpha`, from 0 to :math:`2\pi`
        kind (str):                         'upgoing' for :math:`g^+` and 'downgoing' for :math:`g^-` type
                                            expansions 
        reference_point (list or tuple):    [x, y, z]-coordinates of point relative to which the plane waves are 
                                            defined.
        lower_z (float):                    the expansion is valid on and above that z-coordinate
        upper_z (float):                    the expansion is valid below that z-coordinate
        

    Attributes:
        coefficients (numpy ndarray): coefficients[j, k, l] contains 
        :math:`g^\pm_{j}(\kappa_{k}, \alpha_{l})`
    """
    def __init__(self, k, k_parallel, azimuthal_angles, kind=None, reference_point=None, lower_z=-np.inf,
                 upper_z=np.inf):
        FieldExpansion.__init__(self)
        self.k = k
        self.k_parallel = np.array(k_parallel, ndmin=1)
        self.azimuthal_angles = np.array(azimuthal_angles, ndmin=1)
        self.kind = kind  # 'upgoing' or 'downgoing'
        self.reference_point = reference_point
        self.lower_z = lower_z
        self.upper_z = upper_z

        # The coefficients :math:`g^\pm_{j}(\kappa,\alpha) are represented as a 3-dimensional numpy.ndarray.
        # The indices are:
        # - polarization (0=TE, 1=TM)
        # - index of the kappa dimension
        # - index of the alpha dimension
        self.coefficients = np.zeros((2, len(self.k_parallel), len(self.azimuthal_angles)), dtype=complex)

    def valid(self, x, y, z):
        """Test if points are in definition range of the expansion.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            definition domain.
        """
        vld = np.logical_and(z >= self.lower_z, z < self.upper_z)
        vld_custom = FieldExpansion.valid(self, x, y, z)
        return np.logical_and(vld, vld_custom)

    def diverging(self, x, y, z):
        """Test if points are in domain where expansion could diverge.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            divergence domain.
        """
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
            kz = smuthi.fields.k_z(k_parallel=self.k_parallel, k=self.k)
        elif self.kind == 'downgoing':
            kz = -smuthi.fields.k_z(k_parallel=self.k_parallel, k=self.k)
        else:
            raise ValueError('pwe kind undefined')
        return kz

    def k_z_grid(self):
        if self.kind == 'upgoing':
            kz = smuthi.fields.k_z(k_parallel=self.k_parallel_grid(), k=self.k)
        elif self.kind == 'downgoing':
            kz = -smuthi.fields.k_z(k_parallel=self.k_parallel_grid(), k=self.k)
        else:
            raise ValueError('pwe type undefined')
        return kz

    def compatible(self, other):
        """Check if two plane wave expansions are compatible in the sense that 
        they can be added coefficient-wise

        Args:
            other (FieldExpansion):  expansion object to add to this object

        Returns:
            bool (true if compatible, false else)
        """
        return (type(other).__name__=="PlaneWaveExpansion" and np.isclose(self.k, other.k)
                and all(np.isclose(self.k_parallel, other.k_parallel))
                and all(np.isclose(self.azimuthal_angles, other.azimuthal_angles)) and self.kind == other.kind
                and self.reference_point == other.reference_point)

    def __add__(self, other):
        if not self.compatible(other):
            raise ValueError('Plane wave expansion are inconsistent.')
        pwe_sum = PlaneWaveExpansion(k=self.k, k_parallel=self.k_parallel, azimuthal_angles=self.azimuthal_angles,
                                     kind=self.kind, reference_point=self.reference_point,
                                     lower_z=max(self.lower_z, other.lower_z),
                                     upper_z=min(self.upper_z, other.upper_z))
        pwe_sum.coefficients = self.coefficients + other.coefficients
        return pwe_sum
    
    def electric_field(self, x, y, z, chunksize=50):
        """Evaluate electric field.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
            chunksize (int):      number of field points that are simultaneously 
                                  evaluated when running in CPU mode
         
        Returns:
            Tuple of (E_x, E_y, E_z) numpy.ndarray objects with the Cartesian 
            coordinates of complex electric field.
        """
        # todo: replace chunksize argument by automatic estimate (considering available RAM)
        ex = np.zeros(x.shape, dtype=complex)
        ey = np.zeros(x.shape, dtype=complex)
        ez = np.zeros(x.shape, dtype=complex)

        abc = self.valid(x, y, z)
        xr = x[self.valid(x, y, z)] - self.reference_point[0]
        yr = y[self.valid(x, y, z)] - self.reference_point[1]
        zr = z[self.valid(x, y, z)] - self.reference_point[2]
        
        if cu.use_gpu and xr.size and len(self.k_parallel) > 1:  # run calculations on gpu
            

            re_k_d = cu.gpuarray.to_gpu(np.array(self.k).real.astype(np.float32))
            im_k_d = cu.gpuarray.to_gpu(np.array(self.k).imag.astype(np.float32))

            re_kp_d = cu.gpuarray.to_gpu(self.k_parallel.real.astype(np.float32))
            im_kp_d = cu.gpuarray.to_gpu(self.k_parallel.imag.astype(np.float32))

            re_kz_d = cu.gpuarray.to_gpu(self.k_z().real.astype(np.float32))
            im_kz_d = cu.gpuarray.to_gpu(self.k_z().imag.astype(np.float32))

            alpha_d = cu.gpuarray.to_gpu(self.azimuthal_angles.astype(np.float32))

            xr_d = cu.gpuarray.to_gpu(xr.astype(np.float32))
            yr_d = cu.gpuarray.to_gpu(yr.astype(np.float32))
            zr_d = cu.gpuarray.to_gpu(zr.astype(np.float32))

            re_g_te_d = cu.gpuarray.to_gpu(self.coefficients[0, :, :].real.astype(np.float32))
            im_g_te_d = cu.gpuarray.to_gpu(self.coefficients[0, :, :].imag.astype(np.float32))
            re_g_tm_d = cu.gpuarray.to_gpu(self.coefficients[1, :, :].real.astype(np.float32))
            im_g_tm_d = cu.gpuarray.to_gpu(self.coefficients[1, :, :].imag.astype(np.float32))

            re_e_x_d = cu.gpuarray.to_gpu(np.zeros(xr.shape, dtype=np.float32))
            im_e_x_d = cu.gpuarray.to_gpu(np.zeros(xr.shape, dtype=np.float32))
            re_e_y_d = cu.gpuarray.to_gpu(np.zeros(xr.shape, dtype=np.float32))
            im_e_y_d = cu.gpuarray.to_gpu(np.zeros(xr.shape, dtype=np.float32))
            re_e_z_d = cu.gpuarray.to_gpu(np.zeros(xr.shape, dtype=np.float32))
            im_e_z_d = cu.gpuarray.to_gpu(np.zeros(xr.shape, dtype=np.float32))

            kernel_source = cu_src.pwe_electric_field_evaluation_code%(xr.size, len(self.k_parallel),
                                                                   len(self.azimuthal_angles), (1/self.k).real,
                                                                   (1/self.k).imag)

            kernel_function = cu.SourceModule(kernel_source).get_function("electric_field")
            cuda_blocksize = 128
            cuda_gridsize = (xr.size + cuda_blocksize - 1) // cuda_blocksize
            
            kernel_function(re_kp_d, im_kp_d, re_kz_d, im_kz_d, alpha_d, xr_d, yr_d, zr_d, re_g_te_d, im_g_te_d,
                            re_g_tm_d, im_g_tm_d, re_e_x_d, im_e_x_d, re_e_y_d, im_e_y_d, re_e_z_d, im_e_z_d,
                            block=(cuda_blocksize,1,1), grid=(cuda_gridsize,1))
            
            ex[self.valid(x, y, z)] = re_e_x_d.get() + 1j * im_e_x_d.get()
            ey[self.valid(x, y, z)] = re_e_y_d.get() + 1j * im_e_y_d.get()
            ez[self.valid(x, y, z)] = re_e_z_d.get() + 1j * im_e_z_d.get()
            
        else:  # run calculations on cpu
            kpgrid = self.k_parallel_grid()
            agrid = self.azimuthal_angle_grid()
            kx = kpgrid * np.cos(agrid)
            ky = kpgrid * np.sin(agrid)
            kz = self.k_z_grid()

            e_x_flat = np.zeros(xr.size, dtype=np.complex64)
            e_y_flat = np.zeros(xr.size, dtype=np.complex64)
            e_z_flat = np.zeros(xr.size, dtype=np.complex64)

            for i_chunk in range(math.ceil(xr.size / chunksize)):
                chunk_idcs = range(i_chunk * chunksize, min((i_chunk + 1) * chunksize, xr.size))
                xr_chunk = xr.flatten()[chunk_idcs]
                yr_chunk = yr.flatten()[chunk_idcs]
                zr_chunk = zr.flatten()[chunk_idcs]

                kr = np.zeros((len(xr_chunk), len(self.k_parallel), 
                               len(self.azimuthal_angles)), dtype=np.complex64)
                kr += np.tensordot(xr_chunk, kx, axes=0)
                kr += np.tensordot(yr_chunk, ky, axes=0)
                kr += np.tensordot(zr_chunk, kz, axes=0)

                eikr = np.exp(1j * kr)
        
                integrand_x = np.zeros((len(xr_chunk), len(self.k_parallel), len(self.azimuthal_angles)),
                                       dtype=np.complex64)
                integrand_y = np.zeros((len(yr_chunk), len(self.k_parallel), len(self.azimuthal_angles)),
                                       dtype=np.complex64)
                integrand_z = np.zeros((len(zr_chunk), len(self.k_parallel), len(self.azimuthal_angles)),
                                       dtype=np.complex64)

                # pol=0
                integrand_x += (-np.sin(agrid) * self.coefficients[0, :, :])[None, :, :] * eikr
                integrand_y += (np.cos(agrid) * self.coefficients[0, :, :])[None, :, :] * eikr
                # pol=1
                integrand_x += (np.cos(agrid) * kz / self.k * self.coefficients[1, :, :])[None, :, :] * eikr
                integrand_y += (np.sin(agrid) * kz / self.k * self.coefficients[1, :, :])[None, :, :] * eikr
                integrand_z += (-kpgrid / self.k * self.coefficients[1, :, :])[None, :, :] * eikr

                if len(self.k_parallel) > 1:
                    e_x_flat[chunk_idcs] = np.trapz(np.trapz(integrand_x, self.azimuthal_angles)
                                                    * self.k_parallel, self.k_parallel)
                    e_y_flat[chunk_idcs] = np.trapz(np.trapz(integrand_y, self.azimuthal_angles)
                                                    * self.k_parallel, self.k_parallel)
                    e_z_flat[chunk_idcs] = np.trapz(np.trapz(integrand_z, self.azimuthal_angles)
                                                    * self.k_parallel, self.k_parallel)
                else:
                    e_x_flat[chunk_idcs] = np.squeeze(integrand_x)
                    e_y_flat[chunk_idcs] = np.squeeze(integrand_y)
                    e_z_flat[chunk_idcs] = np.squeeze(integrand_z)

            ex[self.valid(x, y, z)] = e_x_flat.reshape(xr.shape)
            ey[self.valid(x, y, z)] = e_y_flat.reshape(xr.shape)
            ez[self.valid(x, y, z)] = e_z_flat.reshape(xr.shape)

        return ex, ey, ez
