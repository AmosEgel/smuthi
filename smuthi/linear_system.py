import smuthi.t_matrix as tmt
import smuthi.particle_coupling as coup
import smuthi.field_expansion as fldex
import smuthi.coordinates as coord
import numpy as np
import sys
import scipy.linalg
import scipy.interpolate
import scipy.sparse.linalg
try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule
    import pycuda.cumath
    use_gpu = True
except ImportError:
    import warnings
    warnings.warn("Unable to import PyCuda module - fall back to CPU mode")
    use_gpu = False
from tqdm import tqdm
import time

iter_num = 0


class LinearSystem:
    """Manage the assembly and solution of the linear system of equations.

    Args:
        particle_list (list):   List of smuthi.particles.Particle objects
        initial_field (smuthi.initial_field.InitialField):   Initial field object
        layer_system (smuthi.layers.LayerSystem):   Stratified medium
        k_parallel (numpy.ndarray or str): in-plane wavenumber. If 'default', use smuthi.coord.default_k_parallel
        solver_type (str):  What solver to use? Options: 'LU' for LU factorization, 'gmres' for GMRES iterative solver
        store_coupling_matrix (bool):   If True (default), the coupling matrix is stored. Otherwise it is recomputed on
                                        the fly during each iteration of the solver.
        coupling_matrix_lookup_resolution (float or None): If type float, compute particle coupling by interpolation of
                                                           a lookup table with that spacial resolution. If None
                                                           (default), don't use a lookup table but compute the coupling
                                                           directly. This is more suitable for a small particle number.
    """
    def __init__(self, particle_list, initial_field, layer_system, k_parallel='default', solver_type='LU', 
                 store_coupling_matrix=True, coupling_matrix_lookup_resolution=None):
        
        self.k_parallel = k_parallel
        self.solver_type = solver_type
        
        self.particle_list = particle_list
        self.initial_field = initial_field
        self.layer_system = layer_system

        #sys.stdout.write("Compute initial field coefficients ... \n")
        #sys.stdout.flush()
        for particle in tqdm(particle_list, desc='Initial field coefficients', file=sys.stdout,
                             bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}'):
            particle.initial_field = initial_field.spherical_wave_expansion(particle, layer_system)
        
        for particle in tqdm(particle_list, desc='T-matrices                ', file=sys.stdout,
                             bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}'):
            niS = layer_system.refractive_indices[layer_system.layer_number(particle.position[2])]
            particle.t_matrix = tmt.t_matrix(initial_field.vacuum_wavelength, niS, particle)
        
        self.coupling_matrix = CouplingMatrix(vacuum_wavelength=initial_field.vacuum_wavelength, 
                                              particle_list=particle_list, layer_system=layer_system, 
                                              k_parallel=self.k_parallel, store_matrix=store_coupling_matrix, 
                                              lookup_resolution=coupling_matrix_lookup_resolution)
        
        sys.stdout.flush()
        self.t_matrix = TMatrix(particle_list=particle_list)
        self.master_matrix = MasterMatrix(t_matrix=self.t_matrix, coupling_matrix=self.coupling_matrix)
        

    def solve(self):
        """Compute scattered field coefficients and store them in the particles' spherical wave expansion objects."""
        sys.stdout.flush()
        if len(self.particle_list) > 0:
            if self.solver_type == 'LU':
                sys.stdout.write('Solve (LU decomposition)  : ...')
                if not hasattr(self.master_matrix.linear_operator, 'A'):
                    raise ValueError('LU factorization only possible with the option "store coupling matrix".')
                if not hasattr(self.master_matrix, 'LU_piv'):
                    lu, piv = scipy.linalg.lu_factor(self.master_matrix.linear_operator.A, overwrite_a=False)
                    self.master_matrix.LU_piv = (lu, piv)
                b = scipy.linalg.lu_solve(self.master_matrix.LU_piv, self.t_matrix.right_hand_side())
                sys.stdout.write(' done\n')
            elif self.solver_type == 'gmres':
                rhs = self.t_matrix.right_hand_side()
                start_time = time.time()
                def status_msg(rk):
                    global iter_num
                    iter_msg = ('Solve (GMRES)             : Iter ' + str(iter_num) + ' | Rel. residual: '
                                + "{:.2e}".format(np.linalg.norm(rk) / np.linalg.norm(rhs)) + ' | elapsed: ' 
                                + str(int(time.time() - start_time)) + 's')
                    sys.stdout.write('\r' + iter_msg)
                    iter_num += 1
                b, info = scipy.sparse.linalg.gmres(self.master_matrix.linear_operator, rhs, rhs, callback=status_msg)
                sys.stdout.write('\n')
            else:
                raise ValueError('This solver type is currently not implemented.')

        for iS, particle in enumerate(self.particle_list):
            i_iS = self.layer_system.layer_number(particle.position[2])
            n_iS = self.layer_system.refractive_indices[i_iS]
            k = coord.angular_frequency(self.initial_field.vacuum_wavelength) * n_iS
            loz, upz = self.layer_system.lower_zlimit(i_iS), self.layer_system.upper_zlimit(i_iS)
            particle.scattered_field = fldex.SphericalWaveExpansion(k=k, l_max=particle.l_max, m_max=particle.m_max,
                                                                    kind='outgoing', reference_point=particle.position,
                                                                    lower_z=loz, upper_z=upz)
            particle.scattered_field.coefficients = b[self.master_matrix.index_block(iS)]


class SystemMatrix:
    """A system matrix is an abstract linear operator that operates on a system coefficient vector, i.e. a vector 
    :math:`c = c_{\tau,l,m}^i`, where :math:`(\tau, l, m)` are the multipole indices and :math:`i` indicates the 
    particle number.
    """
    def __init__(self, particle_list):
        self.particle_list = particle_list
        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in self.particle_list]
        self.shape = (sum(blocksizes), sum(blocksizes))
    
    def index_block(self, i):
        """
        Args:
            i (int): number of particle

        Returns:
            indices that correspond to the coefficients for that particle
        """
        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in self.particle_list[:(i + 1)]]
        return range(sum(blocksizes[:i]), sum(blocksizes))
    
    def index(self, i, tau, l, m):
        """
        Args:
            i (int):    particle number
            tau (int):    spherical polarization index
            l (int):    multipole degree
            m (int):    multipole order
        
        Returns:
            Position in a system vector that corresponds to the :math:`(\tau, l, m)` coefficient of the i-th particle. 
        """
        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in self.particle_list[:i]]
        return sum(blocksizes) + fldex.multi_to_single_index(tau, l, m, self.particle_list[i].l_max, 
                                                             self.particle_list[i].m_max)


class CouplingMatrix(SystemMatrix):
    """The direct and layer mediated particle coupling coefficients represented as a linear operator.

    Args:
        vacuum_wavelength (float):  Vacuum wavelength in length units
        particle_list (list):   List of smuthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem):   Stratified medium
        k_parallell (numpy.ndarray or str): In-plane wavenumber. If 'default', use smuthi.coordinates.default_k_parallel
        store_matrix (bool):    If True (default), the coupling matrix is stored. Otherwise it is recomputed on the fly
                                during each iteration of the solver.
        lookup_resolution (float or None): If type float, compute particle coupling by interpolation of a lookup table
                                            with that spacial resolution. If None (default), don't use a lookup table
                                            but compute the coupling directly. This is more suitable for a small
                                            particle number.
    """
    def __init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel='default', store_matrix=True,
                 lookup_resolution=None):
        
        SystemMatrix.__init__(self, particle_list)
        
        if lookup_resolution is not None:
            x_array = np.array([particle.position[0] for particle in particle_list])
            y_array = np.array([particle.position[1] for particle in particle_list])
            z_list = [particle.position[2] for particle in particle_list]
            if z_list.count(z_list[0]) == len(z_list):
                self.particle_rho_array = np.sqrt((x_array[:, None] - x_array[None, :])**2
                                                  + (y_array[:, None] - y_array[None, :])**2)
                self.particle_phi_array = np.arctan2(y_array[:, None] - y_array[None, :],
                                                     x_array[:, None] - x_array[None, :])
                self.lookup_table, self.radial_distance_array = coup.radial_coupling_lookup_table(
                    vacuum_wavelength=vacuum_wavelength, particle_list=particle_list, layer_system=layer_system,
                    k_parallel=k_parallel, resolution=lookup_resolution)
                self.lookup_resolution = lookup_resolution

                self.l_max = max([particle.l_max for particle in particle_list])
                self.m_max = max([particle.m_max for particle in particle_list])
                self.n_max = fldex.blocksize(self.l_max, self.m_max)
                
                self.system_vector_index_list = [[] for i in range(self.n_max)]  # contains for each n all positions in the large system arrays that correspond to n
                self.particle_number_list = [[] for i in range(self.n_max)]  # same size as system_vector_index_list, contains the according particle numbers
                self.m_list = [None for i in range(self.n_max)]
                for i, particle in enumerate(particle_list):
                    for m in range(-particle.m_max, particle.m_max + 1):
                        for l in range(max(1, abs(m)), particle.l_max + 1):
                            for tau in range(2):
                                n_lookup = fldex.multi_to_single_index(tau=tau, l=l, m=m, l_max=self.l_max, 
                                                                       m_max=self.m_max)
                                self.system_vector_index_list[n_lookup].append(self.index(i, tau, l, m))
                                self.particle_number_list[n_lookup].append(i)
                                self.m_list[n_lookup] = m
                for n in range(self.n_max):
                    self.system_vector_index_list[n] = np.array(self.system_vector_index_list[n])
                    self.particle_number_list[n] = np.array(self.particle_number_list[n])
            else:
                raise NotImplementedError('3D lookup not yet implemented')

        if store_matrix:
            if lookup_resolution is None:
                coup_mat = np.zeros(self.shape, dtype=complex)
                for s1, particle1 in enumerate(tqdm(particle_list, desc='Particle coupling matrix  ', file=sys.stdout,
                                                    bar_format='{l_bar}{bar}| elapsed: {elapsed} '
                                                    'remaining: {remaining}')):
                    idx1 = np.array(self.index_block(s1))[:, None]
                    for s2, particle2 in enumerate(particle_list):
                        idx2 = self.index_block(s2)
                        coup_mat[idx1, idx2] = (coup.layer_mediated_coupling_block(vacuum_wavelength, particle1,
                                                                                   particle2, layer_system, k_parallel)
                                                + coup.direct_coupling_block(vacuum_wavelength, particle1, particle2,
                                                                             layer_system))
            else:
                raise NotImplementedError('lookups not yet implemtned')
            self.linear_operator = scipy.sparse.linalg.aslinearoperator(coup_mat)
        else:
            if use_gpu:
                re_lookup_d = [[None for n2 in range(self.n_max)] for n1 in range(self.n_max)]
                im_lookup_d = [[None for n2 in range(self.n_max)] for n1 in range(self.n_max)]
                for n1 in range(self.n_max):
                    for n2 in range(self.n_max):
                        re_lookup_d[n1][n2] = gpuarray.to_gpu(np.float32(self.lookup_table[n1][n2].real))
                        im_lookup_d[n1][n2] = gpuarray.to_gpu(np.float32(self.lookup_table[n1][n2].imag))
                
                normalized_rho_array = np.float32(self.particle_rho_array / self.lookup_resolution)
                interp1_cuda = SourceModule("""
                __global__ void coupling_kernel(const float *rho, const float *phi, const int N, 
                                                const float dm, const float *re_lookup, 
                                                const float *im_lookup, float  *re_result, float  *im_result)
                {
                    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                    if(x >= N) return;
                    const int idx = (int) floor(rho[x]);
                    const float w = rho[x] - floor(rho[x]);
                    const float re_si = re_lookup[idx] + w * (re_lookup[idx + 1] - re_lookup[idx]);
                    const float im_si = im_lookup[idx] + w * (im_lookup[idx + 1] - im_lookup[idx]);
                    const float re_eimphi = cosf(dm * phi[x]);
                    const float im_eimphi = sinf(dm * phi[x]);
                    re_result[x] = re_eimphi * re_si - im_eimphi * im_si;
                    im_result[x] = im_eimphi * re_si + re_eimphi * im_si;
                }
                """).get_function("coupling_kernel")
                
                def evaluate_coupling_matrix(in_vec):
                    out_vec = np.zeros(shape=in_vec.shape, dtype=complex)
                    for n1 in range(self.n_max):
                        i1 = self.particle_number_list[n1]
                        idx1 = self.system_vector_index_list[n1]
                        m1 = self.m_list[n1]
                        for n2 in range(self.n_max):
                            i2 = self.particle_number_list[n2]
                            idx2 = self.system_vector_index_list[n2]
                            m2 = self.m_list[n2]
                            
                            rho_array_d = gpuarray.to_gpu(normalized_rho_array[i1[:, None], i2[None, :]])
                            phi_array_d = gpuarray.to_gpu(np.float32(self.particle_phi_array[i1[:, None], i2[None, :]]))
                            
                            re_w_d = gpuarray.empty_like(rho_array_d)
                            im_w_d = gpuarray.empty_like(rho_array_d)
                            
                            dm = np.float32(m2 - m1)
                            
                            blocksize = 128
                            gridsize = int(np.ceil(len(i1) * len(i2) / blocksize))
                
                            interp1_cuda(rho_array_d.gpudata, phi_array_d.gpudata, np.uint32(len(i1) * len(i2)), dm, 
                                         re_lookup_d[n1][n2].gpudata, im_lookup_d[n1][n2].gpudata, 
                                         re_w_d.gpudata, im_w_d.gpudata, block=(blocksize,1,1), grid=(gridsize,1))
                            w = re_w_d.get() + 1j * im_w_d.get()
                            out_vec[idx1] += w.dot(in_vec[idx2])
                    return out_vec
                    
                    
            
            else:
                lookup = [[None for i in range(self.n_max)] for i2 in range(self.n_max)]
                for n1 in range(self.n_max):
                    for n2 in range(self.n_max):
                        lookup[n1][n2] = scipy.interpolate.interp1d(x=self.radial_distance_array, y=self.lookup_table[n1][n2], 
                                                                    kind='linear', axis=-1, assume_sorted=True)
#                @profile
                def evaluate_coupling_matrix(in_vec):
                    out_vec = np.zeros(shape=in_vec.shape, dtype=complex)
                    for n1 in range(self.n_max):
                        i1 = self.particle_number_list[n1]
                        idx1 = self.system_vector_index_list[n1]
                        m1 = self.m_list[n1]
                        for n2 in range(self.n_max):
                            i2 = self.particle_number_list[n2]
                            idx2 = self.system_vector_index_list[n2]
                            m2 = self.m_list[n2]
                            rho = self.particle_rho_array[i1[:, None], i2[None, :]]
                            phi = self.particle_phi_array[i1[:, None], i2[None, :]]
                            M = lookup[n1][n2](rho)
                            M = M * np.exp(1j * (m2 - m1) * phi)
                            out_vec[idx1] += M.dot(in_vec[idx2])
                    return out_vec
            self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=self.shape, matvec=evaluate_coupling_matrix,
                                                                      dtype=complex)

class TMatrix(SystemMatrix):
    """Collect the particle T-matrices in a global lienear operator.

    Args:
        particle_list (list):   List of smuthi.particles.Particle objects containing a t_matrix attribute.
    """
    def __init__(self, particle_list):
        SystemMatrix.__init__(self, particle_list)
        def apply_t_matrix(vector):
            tv = np.zeros(vector.shape, dtype=complex)
            for i_s, particle in enumerate(particle_list):
                tv[self.index_block(i_s)] = particle.t_matrix.dot(vector[self.index_block(i_s)])
            return tv
        self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=self.shape, matvec=apply_t_matrix, 
                                                                  matmat=apply_t_matrix, dtype=complex)
    
    def right_hand_side(self):
        r"""The right hand side of the linear system is given by :math:`\sum_{\tau l m} T^i_{\tau l m} a^i_{\tau l m }`

        Returns:
            right hand side as a complex numpy.ndarray
        """
        tai = np.zeros(self.shape[0], dtype=complex)
        for i_s, particle in enumerate(self.particle_list):
            tai[self.index_block(i_s)] = particle.t_matrix.dot(particle.initial_field.coefficients)
        return tai

        
class MasterMatrix(SystemMatrix):
    r"""Represent the master matrix :math:`M = 1 - TW` as a linear operator.

    Args:
        t_matrix (SystemTMatrix):    T-matrix object
        coupling_matrix (CouplingMatrix):   Coupling matrix object
    """
    def __init__(self, t_matrix, coupling_matrix):
        SystemMatrix.__init__(self, t_matrix.particle_list)
        if type(coupling_matrix.linear_operator).__name__ == 'MatrixLinearOperator':
            M = (np.eye(self.shape[0], dtype=complex) 
                 - t_matrix.linear_operator.matmat(coupling_matrix.linear_operator.A))
            self.linear_operator = scipy.sparse.linalg.aslinearoperator(M)
        else:
            def apply_master_matrix(vector):
                return vector - t_matrix.linear_operator.dot(coupling_matrix.linear_operator.matvec(vector))                
            self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=self.shape, matvec=apply_master_matrix, 
                                                                      dtype=complex)
        