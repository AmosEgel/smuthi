"""This module contains classes to represent the particle coupling matrix for all particles, i.e., on system level."""

import sys
import tqdm
import time
import numpy as np
import scipy.special
import scipy.sparse.linalg
import smuthi.fields.expansions as fldex
import smuthi.utility.cuda as cu
import smuthi.linear_system as linsys
import smuthi.linear_system.particle_coupling.direct_coupling as dircoup
import smuthi.linear_system.particle_coupling.layer_mediated_coupling as laycoup
import smuthi.linear_system.particle_coupling.prepare_lookup as look
try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule
    import pycuda.cumath
except:
    pass


class CouplingMatrixExplicit(linsys.SystemMatrix):
    """Class for an explicit representation of the coupling matrix. Recommended for small particle numbers.

    Args:
        vacuum_wavelength (float):  Vacuum wavelength in length units
        particle_list (list):   List of smuthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem):   Stratified medium
        k_parallell (numpy.ndarray or str): In-plane wavenumber. If 'default', use smuthi.coordinates.default_k_parallel
    """
    def __init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel='default'):
      
        linsys.SystemMatrix.__init__(self, particle_list)
        coup_mat = np.zeros(self.shape, dtype=complex)
        sys.stdout.write('Coupling matrix memory footprint: ' + look.size_format(coup_mat.nbytes) + '\n')
        sys.stdout.flush()
        for s1, particle1 in enumerate(tqdm(particle_list, desc='Particle coupling matrix  ', file=sys.stdout,
                                            bar_format='{l_bar}{bar}| elapsed: {elapsed} ' 'remaining: {remaining}')):
            idx1 = np.array(self.index_block(s1))[:, None]
            for s2, particle2 in enumerate(particle_list):
                idx2 = self.index_block(s2)
                coup_mat[idx1, idx2] = (laycoup.layer_mediated_coupling_block(vacuum_wavelength, particle1, particle2,
                                                                              layer_system, k_parallel)
                                        + dircoup.direct_coupling_block(vacuum_wavelength, particle1, particle2,
                                                                        layer_system))
        self.linear_operator = scipy.sparse.linalg.aslinearoperator(coup_mat)
      
        
class CouplingMatrixVolumeLookup(linsys.SystemMatrix):
    """Base class for 3D lookup based coupling matrix either on CPU or on GPU (CUDA).
  
    Args:
        vacuum_wavelength (float): vacuum wavelength in length units
        particle_list (list): list of sumthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem): stratified medium
        k_parallel (numpy.ndarray or str): in-plane wavenumber. If 'default', use smuthi.coord.default_k_parallel
        resolution (float or None): spatial resolution of the lookup in the radial direction
    """
    def __init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel='default', resolution=None):
      
        z_list = [particle.position[2] for particle in particle_list]
        is_list = [layer_system.layer_number(z) for z in z_list]
        assert is_list.count(is_list[0]) == len(is_list)  # all particles in same layer?
      
        linsys.SystemMatrix.__init__(self, particle_list)
      
        self.l_max = max([particle.l_max for particle in particle_list])
        self.m_max = max([particle.m_max for particle in particle_list])
        self.blocksize = fldex.blocksize(self.l_max, self.m_max)
        self.resolution = resolution
        lkup = look.volumetric_coupling_lookup_table(vacuum_wavelength=vacuum_wavelength, particle_list=particle_list,
                                                     layer_system=layer_system, k_parallel=k_parallel, 
                                                     resolution=resolution)
        self.lookup_table_plus, self.lookup_table_minus = lkup[0], lkup[1]
        self.rho_array, self.sum_z_array, self.diff_z_array = lkup[2], lkup[3], lkup[4]


class CouplingMatrixVolumeLookupCPU(CouplingMatrixVolumeLookup):
    """Class for 3D lookup based coupling matrix running on CPU. This is used when no suitable GPU device is detected
    or when PyCuda is not installed.
  
    Args:
        vacuum_wavelength (float): vacuum wavelength in length units
        particle_list (list): list of sumthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem): stratified medium
        k_parallel (numpy.ndarray or str): in-plane wavenumber. If 'default', use smuthi.coord.default_k_parallel
        resolution (float or None): spatial resolution of the lookup in the radial direction
        interpolator_kind (str): 'linear' or 'cubic' interpolation
    """
    def __init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel='default', resolution=None,
                 interpolator_kind='cubic'):
        
        if interpolator_kind == 'cubic':
            interpolation_order = 3
        else:
            interpolation_order = 1
            
      
        CouplingMatrixVolumeLookup.__init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel,
                                            resolution)

        x_array = np.array([particle.position[0] for particle in particle_list])
        y_array = np.array([particle.position[1] for particle in particle_list])
        z_array = np.array([particle.position[2] for particle in particle_list])
      
        self.particle_rho_array = np.sqrt((x_array[:, None] - x_array[None, :])**2
                                          + (y_array[:, None] - y_array[None, :])**2)
        self.particle_phi_array = np.arctan2(y_array[:, None] - y_array[None, :], x_array[:, None] - x_array[None, :])
        self.particle_sz_array = z_array[:, None] + z_array[None, :]
        self.particle_dz_array = z_array[:, None] - z_array[None, :]
      
        # contains for each n all positions in the large system arrays that correspond to n:
        self.system_vector_index_list = [[] for i in range(self.blocksize)]

        # same size as system_vector_index_list, contains the according particle numbers:
        self.particle_number_list = [[] for i in range(self.blocksize)]
        self.m_list = [None for i in range(self.blocksize)]
        for i, particle in enumerate(particle_list):
            for m in range(-particle.m_max, particle.m_max + 1):
                for l in range(max(1, abs(m)), particle.l_max + 1):
                    for tau in range(2):
                        n_lookup = fldex.multi_to_single_index(tau=tau, l=l, m=m, l_max=self.l_max, m_max=self.m_max)
                        self.system_vector_index_list[n_lookup].append(self.index(i, tau, l, m))
                        self.particle_number_list[n_lookup].append(i)
                        self.m_list[n_lookup] = m
        for n in range(self.blocksize):
            self.system_vector_index_list[n] = np.array(self.system_vector_index_list[n])
            self.particle_number_list[n] = np.array(self.particle_number_list[n])

        self.lookup_plus_real = [[None for i in range(self.blocksize)] for i2 in range(self.blocksize)]
        self.lookup_plus_imag = [[None for i in range(self.blocksize)] for i2 in range(self.blocksize)]
        self.lookup_minus_real = [[None for i in range(self.blocksize)] for i2 in range(self.blocksize)]
        self.lookup_minus_imag = [[None for i in range(self.blocksize)] for i2 in range(self.blocksize)]
        for n1 in range(self.blocksize):
            for n2 in range(self.blocksize):
                self.lookup_plus_real[n1][n2] = scipy.interpolate.RectBivariateSpline(
                    x=self.rho_array, y=self.sum_z_array, z=self.lookup_table_plus[:, :, n1, n2].real,
                    kx=interpolation_order, ky=interpolation_order)
                self.lookup_plus_imag[n1][n2] = scipy.interpolate.RectBivariateSpline(
                    x=self.rho_array, y=self.sum_z_array, z=self.lookup_table_plus[:, :, n1, n2].imag,
                    kx=interpolation_order, ky=interpolation_order)
                self.lookup_minus_real[n1][n2] = scipy.interpolate.RectBivariateSpline(
                    x=self.rho_array, y=self.diff_z_array, z=self.lookup_table_minus[:, :, n1, n2].real,
                    kx=interpolation_order, ky=interpolation_order)
                self.lookup_minus_imag[n1][n2] = scipy.interpolate.RectBivariateSpline(
                    x=self.rho_array, y=self.diff_z_array, z=self.lookup_table_minus[:, :, n1, n2].imag,
                    kx=interpolation_order, ky=interpolation_order)
        def matvec(in_vec):
            out_vec = np.zeros(shape=in_vec.shape, dtype=complex)
            for n1 in range(self.blocksize):
                i1 = self.particle_number_list[n1]
                idx1 = self.system_vector_index_list[n1]
                m1 = self.m_list[n1]
                for n2 in range(self.blocksize):
                    i2 = self.particle_number_list[n2]
                    idx2 = self.system_vector_index_list[n2]
                    m2 = self.m_list[n2]
                    rho = self.particle_rho_array[i1[:, None], i2[None, :]]
                    phi = self.particle_phi_array[i1[:, None], i2[None, :]]
                    sz = self.particle_sz_array[i1[:, None], i2[None, :]]
                    dz = self.particle_dz_array[i1[:, None], i2[None, :]]
                    Mpl = self.lookup_plus_real[n1][n2].ev(rho, sz) + 1j * self.lookup_plus_imag[n1][n2].ev(rho, sz)
                    Mmn = self.lookup_minus_real[n1][n2].ev(rho, dz) + 1j * self.lookup_minus_imag[n1][n2].ev(rho, dz)
                    M = (Mpl + Mmn) * np.exp(1j * (m2 - m1) * phi)
                    out_vec[idx1] += M.dot(in_vec[idx2])
            return out_vec
        self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=self.shape, matvec=matvec, dtype=complex)


class CouplingMatrixVolumeLookupCUDA(CouplingMatrixVolumeLookup):
    """Class for 3D lookup based coupling matrix running on GPU.

    Args:
        vacuum_wavelength (float): vacuum wavelength in length units
        particle_list (list): list of sumthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem): stratified medium
        k_parallel (numpy.ndarray or str): in-plane wavenumber. If 'default', use smuthi.coord.default_k_parallel
        resolution (float or None): spatial resolution of the lookup in the radial direction
        cuda_blocksize (int): threads per block for cuda call
        interpolator_kind (str): 'linear' (default) or 'cubic' interpolation
    """

    def __init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel='default', resolution=None,
                 cuda_blocksize=None, interpolator_kind='linear'):
        
        if cuda_blocksize is None:
            cuda_blocksize = cu.default_blocksize

        CouplingMatrixVolumeLookup.__init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel,
                                            resolution)

        sys.stdout.write('Prepare CUDA kernel and device lookup data ... ')
        sys.stdout.flush()
        start_time = time.time()
        
        if interpolator_kind == 'linear':
            coupling_source = cu.linear_volume_lookup_source%(self.blocksize, self.shape[0], len(self.sum_z_array), 
                                                              min(self.rho_array), min(self.sum_z_array), 
                                                              min(self.diff_z_array), self.resolution)
        elif interpolator_kind == 'cubic':
            coupling_source = cu.cubic_volume_lookup_source%(self.blocksize, self.shape[0], len(self.sum_z_array),
                                                             min(self.rho_array), min(self.sum_z_array), 
                                                             min(self.diff_z_array), self.resolution)
        
        coupling_function = SourceModule(coupling_source).get_function("coupling_kernel") 
        
        n_lookup_array = np.zeros(self.shape[0], dtype=np.uint32)
        m_particle_array = np.zeros(self.shape[0], dtype=np.float32)
        x_array = np.zeros(self.shape[0], dtype=np.float32)
        y_array = np.zeros(self.shape[0], dtype=np.float32)
        z_array = np.zeros(self.shape[0], dtype=np.float32)

        i_particle = 0
        for i, particle in enumerate(particle_list):
            for m in range(-particle.m_max, particle.m_max + 1):
                for l in range(max(1, abs(m)), particle.l_max + 1):
                    for tau in range(2):
                        i_taulm = fldex.multi_to_single_index(tau, l, m, particle.l_max, particle.m_max)
                        idx = i_particle + i_taulm

                        n_lookup_array[idx] = fldex.multi_to_single_index(tau, l, m, self.l_max, self.m_max)
                        m_particle_array[idx] = m

                        # scale the x and y position to the lookup resolution:
                        x_array[idx] = particle.position[0]
                        y_array[idx] = particle.position[1]
                        z_array[idx] = particle.position[2]

            i_particle += fldex.blocksize(particle.l_max, particle.m_max)

        re_lookup_pl = self.lookup_table_plus.real.astype(dtype=np.float32)
        im_lookup_pl = self.lookup_table_plus.imag.astype(dtype=np.float32)
        re_lookup_mn = self.lookup_table_minus.real.astype(dtype=np.float32)
        im_lookup_mn = self.lookup_table_minus.imag.astype(dtype=np.float32)
        
        # transfer data to gpu
        n_lookup_array_d = gpuarray.to_gpu(n_lookup_array)
        m_particle_array_d = gpuarray.to_gpu(m_particle_array)
        x_array_d = gpuarray.to_gpu(x_array)
        y_array_d = gpuarray.to_gpu(y_array)
        z_array_d = gpuarray.to_gpu(z_array)
        re_lookup_pl_d = gpuarray.to_gpu(re_lookup_pl)
        im_lookup_pl_d = gpuarray.to_gpu(im_lookup_pl)
        re_lookup_mn_d = gpuarray.to_gpu(re_lookup_mn)
        im_lookup_mn_d = gpuarray.to_gpu(im_lookup_mn)

        sys.stdout.write('done | elapsed: ' + str(int(time.time() - start_time)) + 's\n')
        sys.stdout.flush()
        

        cuda_gridsize = (self.shape[0] + cuda_blocksize - 1) // cuda_blocksize

        def matvec(in_vec):
            re_in_vec_d = gpuarray.to_gpu(np.float32(in_vec.real))
            im_in_vec_d = gpuarray.to_gpu(np.float32(in_vec.imag))
            re_result_d = gpuarray.zeros(in_vec.shape, dtype=np.float32)
            im_result_d = gpuarray.zeros(in_vec.shape, dtype=np.float32)
            coupling_function(n_lookup_array_d.gpudata, m_particle_array_d.gpudata, x_array_d.gpudata,
                              y_array_d.gpudata, z_array_d.gpudata, re_lookup_pl_d.gpudata, im_lookup_pl_d.gpudata,
                              re_lookup_mn_d.gpudata, im_lookup_mn_d.gpudata, re_in_vec_d.gpudata, im_in_vec_d.gpudata,
                              re_result_d.gpudata, im_result_d.gpudata, block=(cuda_blocksize, 1, 1),
                              grid=(cuda_gridsize, 1))
            return re_result_d.get() + 1j * im_result_d.get()

        self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=self.shape, matvec=matvec, dtype=complex)


class CouplingMatrixRadialLookup(linsys.SystemMatrix):
    """Base class for radial lookup based coupling matrix either on CPU or on GPU (CUDA).
  
    Args:
        vacuum_wavelength (float): vacuum wavelength in length units
        particle_list (list): list of sumthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem): stratified medium
        k_parallel (numpy.ndarray or str): in-plane wavenumber. If 'default', use smuthi.coord.default_k_parallel
        resolution (float or None): spatial resolution of the lookup in the radial direction
    """
    def __init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel='default', resolution=None):
      
        z_list = [particle.position[2] for particle in particle_list]
        assert z_list.count(z_list[0]) == len(z_list)
      
        linsys.SystemMatrix.__init__(self, particle_list)
       
        self.l_max = max([particle.l_max for particle in particle_list])
        self.m_max = max([particle.m_max for particle in particle_list])
        self.blocksize = fldex.blocksize(self.l_max, self.m_max)
        self.resolution = resolution
        self.lookup_table, self.radial_distance_array = look.radial_coupling_lookup_table(
            vacuum_wavelength=vacuum_wavelength, particle_list=particle_list, layer_system=layer_system,
            k_parallel=k_parallel, resolution=resolution)


class CouplingMatrixRadialLookupCUDA(CouplingMatrixRadialLookup):
    """Radial lookup based coupling matrix either on GPU (CUDA).
  
    Args:
        vacuum_wavelength (float): vacuum wavelength in length units
        particle_list (list): list of sumthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem): stratified medium
        k_parallel (numpy.ndarray or str): in-plane wavenumber. If 'default', use smuthi.coord.default_k_parallel
        resolution (float or None): spatial resolution of the lookup in the radial direction
        cuda_blocksize (int): threads per block when calling CUDA kernel
    """
    def __init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel='default', resolution=None,
                 cuda_blocksize=None, interpolator_kind='linear'):
        
        if cuda_blocksize is None:
            cuda_blocksize = cu.default_blocksize
      
        CouplingMatrixRadialLookup.__init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel, resolution)

        sys.stdout.write('Prepare CUDA kernel and device lookup data ... ')
        sys.stdout.flush()
        start_time = time.time()
                
        if interpolator_kind == 'linear':
            coupling_source = cu.linear_radial_lookup_source%(self.blocksize, self.shape[0], 
                                                              self.radial_distance_array.min(), resolution)
        elif interpolator_kind == 'cubic':
            coupling_source = cu.cubic_radial_lookup_source%(self.blocksize, self.shape[0], 
                                                             self.radial_distance_array.min(), resolution)
            
        coupling_function = SourceModule(coupling_source).get_function("coupling_kernel") 
          
        n_lookup_array = np.zeros(self.shape[0], dtype=np.uint32)
        m_particle_array = np.zeros(self.shape[0], dtype=np.float32)
        x_array = np.zeros(self.shape[0], dtype=np.float32)
        y_array = np.zeros(self.shape[0], dtype=np.float32)

        i_particle = 0
        for i, particle in enumerate(particle_list):
            for m in range(-particle.m_max, particle.m_max + 1):
                for l in range(max(1, abs(m)), particle.l_max + 1):
                    for tau in range(2):

                        #idx = self.index(i, tau, l, m)
                        i_taulm = fldex.multi_to_single_index(tau, l, m, particle.l_max, particle.m_max)
                        idx = i_particle + i_taulm

                        n_lookup_array[idx] = fldex.multi_to_single_index(tau, l, m, self.l_max, self.m_max)
                        m_particle_array[idx] = m

                        # scale the x and y position to the lookup resolution:
                        x_array[idx] = particle.position[0]
                        y_array[idx] = particle.position[1]

            i_particle += fldex.blocksize(particle.l_max, particle.m_max)

        # lookup as numpy array in required shape
        re_lookup = self.lookup_table.real.astype(np.float32)
        im_lookup = self.lookup_table.imag.astype(np.float32)
        
        # transfer data to gpu
        n_lookup_array_d = gpuarray.to_gpu(n_lookup_array)
        m_particle_array_d = gpuarray.to_gpu(m_particle_array)
        x_array_d = gpuarray.to_gpu(x_array)
        y_array_d = gpuarray.to_gpu(y_array)
        re_lookup_d = gpuarray.to_gpu(re_lookup)
        im_lookup_d = gpuarray.to_gpu(im_lookup)
      
        sys.stdout.write('done | elapsed: ' + str(int(time.time() - start_time)) + 's\n')
        sys.stdout.flush()
      
        cuda_gridsize = (self.shape[0] + cuda_blocksize - 1) // cuda_blocksize

        def matvec(in_vec):
            re_in_vec_d = gpuarray.to_gpu(np.float32(in_vec.real))
            im_in_vec_d = gpuarray.to_gpu(np.float32(in_vec.imag))
            re_result_d = gpuarray.zeros(in_vec.shape, dtype=np.float32)
            im_result_d = gpuarray.zeros(in_vec.shape, dtype=np.float32)
            coupling_function(n_lookup_array_d.gpudata, m_particle_array_d.gpudata, x_array_d.gpudata, y_array_d.gpudata,
                              re_lookup_d.gpudata, im_lookup_d.gpudata, re_in_vec_d.gpudata, im_in_vec_d.gpudata, 
                              re_result_d.gpudata, im_result_d.gpudata, block=(cuda_blocksize,1,1), 
                              grid=(cuda_gridsize,1))
            return re_result_d.get() + 1j * im_result_d.get()
       
        self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=self.shape, matvec=matvec, dtype=complex)


class CouplingMatrixRadialLookupCPU(CouplingMatrixRadialLookup):
    """Class for radial lookup based coupling matrix running on CPU. This is used when no suitable GPU device is detected
    or when PyCuda is not installed.
  
    Args:
        vacuum_wavelength (float): vacuum wavelength in length units
        particle_list (list): list of sumthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem): stratified medium
        k_parallel (numpy.ndarray or str): in-plane wavenumber. If 'default', use smuthi.coord.default_k_parallel
        resolution (float or None): spatial resolution of the lookup in the radial direction
        kind (str): interpolation order, e.g. 'linear' or 'cubic'
    """
    def __init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel='default', resolution=None,
                 interpolator_kind='linear'):
      
        z_list = [particle.position[2] for particle in particle_list]
        assert z_list.count(z_list[0]) == len(z_list)
      
        CouplingMatrixRadialLookup.__init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel, resolution)

        x_array = np.array([particle.position[0] for particle in particle_list])
        y_array = np.array([particle.position[1] for particle in particle_list])
      
        self.particle_rho_array = np.sqrt((x_array[:, None] - x_array[None, :])**2
                                          + (y_array[:, None] - y_array[None, :])**2)
        self.particle_phi_array = np.arctan2(y_array[:, None] - y_array[None, :], x_array[:, None] - x_array[None, :])
      
        # contains for each n all positions in the large system arrays that correspond to n:
        self.system_vector_index_list = [[] for i in range(self.blocksize)]

        # same size as system_vector_index_list, contains the according particle numbers:
        self.particle_number_list = [[] for i in range(self.blocksize)]
        self.m_list = [None for i in range(self.blocksize)]
        for i, particle in enumerate(particle_list):
            for m in range(-particle.m_max, particle.m_max + 1):
                for l in range(max(1, abs(m)), particle.l_max + 1):
                    for tau in range(2):
                        n_lookup = fldex.multi_to_single_index(tau=tau, l=l, m=m, l_max=self.l_max, m_max=self.m_max)
                        self.system_vector_index_list[n_lookup].append(self.index(i, tau, l, m))
                        self.particle_number_list[n_lookup].append(i)
                        self.m_list[n_lookup] = m
        for n in range(self.blocksize):
            self.system_vector_index_list[n] = np.array(self.system_vector_index_list[n])
            self.particle_number_list[n] = np.array(self.particle_number_list[n])

        lookup = [[None for i in range(self.blocksize)] for i2 in range(self.blocksize)]
        for n1 in range(self.blocksize):
            for n2 in range(self.blocksize):
                lookup[n1][n2] = scipy.interpolate.interp1d(x=self.radial_distance_array, y=self.lookup_table[:, n1, n2],
                                                            kind=interpolator_kind, axis=-1, assume_sorted=True)

        def matvec(in_vec):
            out_vec = np.zeros(shape=in_vec.shape, dtype=complex)
            for n1 in range(self.blocksize):
                i1 = self.particle_number_list[n1]
                idx1 = self.system_vector_index_list[n1]
                m1 = self.m_list[n1]
                for n2 in range(self.blocksize):
                    i2 = self.particle_number_list[n2]
                    idx2 = self.system_vector_index_list[n2]
                    m2 = self.m_list[n2]
                    rho = self.particle_rho_array[i1[:, None], i2[None, :]]
                    phi = self.particle_phi_array[i1[:, None], i2[None, :]]
                    M = lookup[n1][n2](rho)
                    M = M * np.exp(1j * (m2 - m1) * phi)
                    out_vec[idx1] += M.dot(in_vec[idx2])
            return out_vec
        self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=self.shape, matvec=matvec, dtype=complex)

