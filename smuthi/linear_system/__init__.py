"""This package contains classes and functions to represent the system of 
linear equations that needs to be solved in order to solve the scattering 
problem, see section 3.7 of [Egel 2018 dissertation].

Symbolically, the linear system can be written like

.. math::
    (1 - TW)b = Ta,

where :math:`T` is the transition matrices of the particles, :math:`W` is the 
particle coupling matrix, :math:`b` are the (unknown) coefficients of the 
scattered field in terms of an outgoing spherical wave expansion and :math:`a` 
are the coefficients of the initial field in terms of a regular spherical wave 
expansion."""

import smuthi.linear_system.t_matrix as tmt
from smuthi.linear_system.t_matrix.system_t_matrix import TMatrix
import smuthi.linear_system.particle_coupling.system_coupling_matrix as syscoup
import smuthi.fields.expansions as fldex
import smuthi.fields.coordinates_and_contours as coord
import smuthi.utility.cuda as cu
import numpy as np
import sys
import scipy.linalg
import scipy.interpolate
import scipy.sparse.linalg
from tqdm import tqdm
import time
import warnings

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
                                                           a lookup table with that spacial resolution. A smaller number
                                                           implies higher accuracy and memory footprint.
                                                           If None (default), don't use a lookup table but compute the 
                                                           coupling directly. This is more suitable for a small particle 
                                                           number.
        interpolator_kind (str): interpolation order to be used, e.g. 'linear' or 'cubic'. This argument is ignored if
                                 coupling_matrix_lookup_resolution is None. In general, cubic interpolation is more 
                                 accurate but a bit slower than linear.
                                                           
    """
    def __init__(self, 
                 particle_list, 
                 initial_field, 
                 layer_system, 
                 k_parallel='default', 
                 solver_type='LU', 
                 solver_tolerance=1e-4, 
                 store_coupling_matrix=True, 
                 coupling_matrix_lookup_resolution=None, 
                 interpolator_kind='cubic', 
                 cuda_blocksize=None):
        
        if cuda_blocksize is None:
            cuda_blocksize = cu.default_blocksize
        
        self.particle_list = particle_list
        self.initial_field = initial_field
        self.layer_system = layer_system
        self.k_parallel = k_parallel
        self.solver_type = solver_type
        self.solver_tolerance = solver_tolerance
        self.store_coupling_matrix = store_coupling_matrix
        self.coupling_matrix_lookup_resolution = coupling_matrix_lookup_resolution
        self.interpolator_kind = interpolator_kind
        self.cuda_blocksize = cuda_blocksize

        dummy_matrix = SystemMatrix(self.particle_list)
        sys.stdout.write('Number of unknowns: %i\n' % dummy_matrix.shape[0])

    def prepare(self):
        self.compute_initial_field_coefficients()
        self.compute_t_matrix()
        self.compute_coupling_matrix()
        self.master_matrix = MasterMatrix(t_matrix=self.t_matrix,
                                          coupling_matrix=self.coupling_matrix)

    def compute_initial_field_coefficients(self):
        """Evaluate initial field coefficients."""
        for particle in tqdm(self.particle_list, 
                             desc='Initial field coefficients', 
                             file=sys.stdout,
                             bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}'):
            particle.initial_field = self.initial_field.spherical_wave_expansion(particle, self.layer_system)
        
    def compute_t_matrix(self):
        """Initialize T-matrix object."""
        # make sure that the initialization output appears before the progress bar
        for particle in self.particle_list:
            if type(particle).__name__ != 'Sphere':
                import smuthi.nfmds
                smuthi.nfmds.initialize_binary()
                break

        for particle in tqdm(self.particle_list, 
                             desc='T-matrices                ', 
                             file=sys.stdout,
                             bar_format='{l_bar}{bar}| elapsed: {elapsed} remaining: {remaining}'):
            iS = self.layer_system.layer_number(particle.position[2])
            niS = self.layer_system.refractive_indices[iS]
            particle.t_matrix = tmt.t_matrix(self.initial_field.vacuum_wavelength, niS, particle)
        self.t_matrix = TMatrix(particle_list=self.particle_list)
        
    def compute_coupling_matrix(self):
        """Initialize coupling matrix object."""
        if self.coupling_matrix_lookup_resolution is not None:
            z_list = [particle.position[2] for particle in self.particle_list]
            is_list = [self.layer_system.layer_number(z) for z in z_list]
            if not is_list.count(is_list[0]) == len(is_list):  # all particles in same layer?
                warnings.warn("Particles are not all in same layer. "
                              "Fall back to direct coupling matrix computation (no lookup).")
                self.coupling_matrix_lookup_resolution = None
            if self.store_coupling_matrix:
                warnings.warn("Explicit matrix compuatation using lookup currently not implemented. "
                              "Disabling lookup.")
                self.coupling_matrix_lookup_resolution = None
            else:  # use lookup
                if not self.interpolator_kind in ('linear', 'cubic'):
                    warnings.warn(self.interpolator_kind + ' interpolation not implemented. '
                                  'Use "linear" instead')
                    self.interpolator_kind = 'linear'

                z_list = [particle.position[2] for particle in self.particle_list]
                if z_list.count(z_list[0]) == len(z_list):  # all particles at same height: use radial lookup
                    if cu.use_gpu:
                        sys.stdout.write('Coupling matrix computation by ' + self.interpolator_kind 
                                         + ' interpolation of radial lookup on GPU.\n')
                        sys.stdout.flush()
                        self.coupling_matrix = syscoup.CouplingMatrixRadialLookupCUDA(
                            vacuum_wavelength=self.initial_field.vacuum_wavelength, 
                            particle_list=self.particle_list,
                            layer_system=self.layer_system, 
                            k_parallel=self.k_parallel,
                            resolution=self.coupling_matrix_lookup_resolution, 
                            cuda_blocksize=self.cuda_blocksize,
                            interpolator_kind=self.interpolator_kind)
                    else:
                        sys.stdout.write('Coupling matrix computation by ' + self.interpolator_kind 
                                         + ' interpolation of radial lookup on CPU.\n')
                        sys.stdout.flush()
                        self.coupling_matrix = syscoup.CouplingMatrixRadialLookupCPU(
                            vacuum_wavelength=self.initial_field.vacuum_wavelength, 
                            particle_list=self.particle_list,
                            layer_system=self.layer_system, 
                            k_parallel=self.k_parallel,
                            resolution=self.coupling_matrix_lookup_resolution, 
                            interpolator_kind=self.interpolator_kind)
                else:  # not all particles at same height: use volume lookup
                    if cu.use_gpu:
                        sys.stdout.write('Coupling matrix computation by ' + self.interpolator_kind 
                                         + ' interpolation of 3D lookup on GPU.\n')
                        sys.stdout.flush()
                        self.coupling_matrix = syscoup.CouplingMatrixVolumeLookupCUDA(
                            vacuum_wavelength=self.initial_field.vacuum_wavelength, 
                            particle_list=self.particle_list,
                            layer_system=self.layer_system, 
                            k_parallel=self.k_parallel,
                            resolution=self.coupling_matrix_lookup_resolution, 
                            interpolator_kind=self.interpolator_kind)
                    else:
                        sys.stdout.write('Coupling matrix computation by ' + self.interpolator_kind 
                                         + ' interpolation of 3D lookup on CPU.\n')

                        sys.stdout.flush()
                        self.coupling_matrix = syscoup.CouplingMatrixVolumeLookupCPU(
                            vacuum_wavelength=self.initial_field.vacuum_wavelength, 
                            particle_list=self.particle_list,
                            layer_system=self.layer_system, 
                            k_parallel=self.k_parallel,
                            resolution=self.coupling_matrix_lookup_resolution, 
                            interpolator_kind=self.interpolator_kind)

        if self.coupling_matrix_lookup_resolution is None:
            if not self.store_coupling_matrix:
                warnings.warn("With lookup disabled, coupling matrix needs to be stored.")
                self.store_coupling_matrix = True
            sys.stdout.write('Explicit coupling matrix computation on CPU.\n')
            sys.stdout.flush()
            self.coupling_matrix = syscoup.CouplingMatrixExplicit(vacuum_wavelength=self.initial_field.vacuum_wavelength,
                                                                  particle_list=self.particle_list,
                                                                  layer_system=self.layer_system,
                                                                  k_parallel=self.k_parallel)
      
    def solve(self):
        """Compute scattered field coefficients and store them 
        in the particles' spherical wave expansion objects."""
        if len(self.particle_list) > 0:
            if self.solver_type == 'LU':
                sys.stdout.write('Solve (LU decomposition)  : ...')
                if not hasattr(self.master_matrix.linear_operator, 'A'):
                    raise ValueError('LU factorization only possible '
                                     'with the option "store coupling matrix".')
                if not hasattr(self.master_matrix, 'LU_piv'):
                    lu, piv = scipy.linalg.lu_factor(self.master_matrix.linear_operator.A, 
                                                     overwrite_a=False)
                    self.master_matrix.LU_piv = (lu, piv)
                b = scipy.linalg.lu_solve(self.master_matrix.LU_piv, 
                                          self.t_matrix.right_hand_side())
                sys.stdout.write(' done\n')
                sys.stdout.flush()
            elif self.solver_type == 'gmres':
                rhs = self.t_matrix.right_hand_side()
                start_time = time.time()
                def status_msg(rk):
                    global iter_num
                    iter_msg = ('Solve (GMRES)             : Iter ' + str(iter_num) 
                                + ' | Rel. residual: '
                                + "{:.2e}".format(np.linalg.norm(rk)) 
                                + ' | elapsed: ' + str(int(time.time() - start_time)) + 's')
                    sys.stdout.write('\r' + iter_msg)
                    iter_num += 1
                global iter_num
                iter_num = 0
                b, info = scipy.sparse.linalg.gmres(self.master_matrix.linear_operator, rhs, rhs, 
                                                    tol=self.solver_tolerance, callback=status_msg)
#                sys.stdout.write('\n')
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
    r"""A system matrix is an abstract linear operator that operates on a 
    system coefficient vector, i.e. a vector :math:`c = c_{\tau,l,m}^i`, where 
    :math:`(\tau, l, m)` are the multipole indices and :math:`i` indicates the
    particle number. In other words, if we have a spherical wave expansion for 
    each particle, and write all the expansion coefficients of these expansions
    into one (long) array, what we get is a system vector.
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
        r"""
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


class MasterMatrix(SystemMatrix):
    r"""Represent the master matrix :math:`M = 1 - TW` as a linear operator.

    Args:
        t_matrix (SystemMatrix):          System T-matrix
        coupling_matrix (SystemMatrix):   System coupling matrix
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
