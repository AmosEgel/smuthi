import smuthi.t_matrix as tmt
import smuthi.particle_coupling as coup
import smuthi.field_expansion as fldex
import smuthi.coordinates as coord
import numpy as np
import sys
import scipy.linalg
import scipy.sparse.linalg


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

        sys.stdout.write("Compute initial field coefficients ... ")
        sys.stdout.flush()
        for particle in particle_list:
            particle.initial_field = initial_field.spherical_wave_expansion(particle, layer_system)
        sys.stdout.write("done. \n")

        sys.stdout.write("Compute T-matrices ... ")
        sys.stdout.flush()
        for particle in particle_list:
            niS = layer_system.refractive_indices[layer_system.layer_number(particle.position[2])]
            particle.t_matrix = tmt.t_matrix(initial_field.vacuum_wavelength, niS, particle)
        sys.stdout.write("done. \n")

        sys.stdout.write("Prepare particle coupling ... ")
        sys.stdout.flush()
        self.coupling_matrix = CouplingMatrix(vacuum_wavelength=initial_field.vacuum_wavelength, 
                                              particle_list=particle_list, layer_system=layer_system, 
                                              k_parallel=self.k_parallel, store_matrix=store_coupling_matrix, 
                                              lookup_resolution=coupling_matrix_lookup_resolution)
        sys.stdout.write("done. \n")
        
        sys.stdout.write("Prepare master matrix ... ")
        sys.stdout.flush()
        self.system_t_matrix = SystemTMatrix(particle_list=particle_list)
        self.master_matrix = MasterMatrix(system_t_matrix=self.system_t_matrix, coupling_matrix=self.coupling_matrix)
        sys.stdout.write("done. \n")

    def number_of_unknowns(self):
        """
        Returns:
            total number of complex unknowns parameterizing the scattered field (int)
        """
        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in self.particle_list]
        return sum(blocksizes)

    def index_block(self, i_s):
        """
        Args:
            i_s (int): number of particle

        Returns:
            indices that correspond to the coefficients for that particle
        """
        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in self.particle_list]
        return range(sum(blocksizes[:i_s]), sum(blocksizes[:(i_s + 1)]))

    def solve(self):
        """Compute scattered field coefficients and store them in the particles' spherical wave expansion objects."""
        sys.stdout.write("Solve linear system ... ")
        sys.stdout.flush()
        if len(self.particle_list) > 0:
            if self.solver_type == 'LU':
                if not hasattr(self.master_matrix.linear_operator, 'A'):
                    raise ValueError('LU factorization only possible with the option "store coupling matrix".')
                if not hasattr(self.master_matrix, 'LU_piv'):
                    lu, piv = scipy.linalg.lu_factor(self.master_matrix.linear_operator.A, overwrite_a=False)
                    self.master_matrix.LU_piv = (lu, piv)
                b = scipy.linalg.lu_solve(self.master_matrix.LU_piv, self.right_hand_side())
            elif self.solver_type == 'gmres':
                b, info = scipy.sparse.linalg.gmres(self.master_matrix.linear_operator, self.right_hand_side(),
                                                    self.right_hand_side())
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
            particle.scattered_field.coefficients = b[self.index_block(iS)]

        sys.stdout.write("done. \n")

    def right_hand_side(self):
        r"""The right hand side of the linear system is given by :math:`\sum_{\tau l m} T^i_{\tau l m} a^i_{\tau l m }`

        Returns:
            right hand side as a complex numpy.ndarray
        """
        tai = np.zeros(self.number_of_unknowns(), dtype=complex)
        for iS, particle in enumerate(self.particle_list):
            tai[self.index_block(iS)] = particle.t_matrix.dot(particle.initial_field.coefficients)
        return tai


class CouplingMatrix:
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

        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in particle_list]

        if lookup_resolution is not None:
            x_array = np.array([particle.position[0] for particle in particle_list])
            y_array = np.array([particle.position[1] for particle in particle_list])
            z_list = [particle.position[2] for particle in particle_list]
            if z_list.count(z_list[0]) == len(z_list):
                sys.stdout.write("Initialize radial particle coupling lookup ... ")
                sys.stdout.flush()
                self.particle_rho_array = np.sqrt((x_array[:, None] - x_array[None, :])**2
                                                  + (y_array[:, None] - y_array[None, :])**2)
                self.particle_phi_array = np.arctan2(y_array[:, None] - y_array[None, :],
                                                     x_array[:, None] - x_array[None, :])
                self.lookup = coup.radial_coupling_lookup(vacuum_wavelength=vacuum_wavelength,
                                                          particle_list=particle_list, layer_system=layer_system,
                                                          k_parallel=k_parallel, resolution=lookup_resolution)

                self.l_max = max([particle.l_max for particle in particle_list])
                self.m_max = max([particle.m_max for particle in particle_list])
                self.n_max = fldex.blocksize(self.l_max, self.m_max)
                self.global_index_list = [[] for i in range(self.n_max)]  # contains for each n all positions in the large system arrays that correspond to n
                self.particle_index_list = [[] for i in range(self.n_max)]  # same size as global_index_list, contains the according particle numbers
                self.m_list = [None for i in range(self.n_max)]
                counter = 0
                for i, particle in enumerate(particle_list):
                    for m in range(-particle.m_max, particle.m_max + 1):
                        for l in range(max(1, abs(m)), particle.l_max + 1):
                            for tau in range(2):
                                n = fldex.multi_to_single_index(tau=tau, l=l, m=m, l_max=particle.l_max,
                                                                m_max=particle.m_max)
                                self.global_index_list[n].append(counter)
                                self.particle_index_list[n].append(i)
                                self.m_list[n] = m
                                counter += 1
                for n in range(self.n_max):
                    self.global_index_list[n] = np.array(self.global_index_list[n])
                    self.particle_index_list[n] = np.array(self.particle_index_list[n])
            else:
                raise NotImplementedError('3D lookup not yet implemented')

        if store_matrix:
            if lookup_resolution is None:
                coup_mat = np.zeros((sum(blocksizes), sum(blocksizes)), dtype=complex)
                for s1, particle1 in enumerate(particle_list):
                    idx1 = np.array(range(sum(blocksizes[:s1]), sum(blocksizes[:s1+1])))[:, None]
                    for s2, particle2 in enumerate(particle_list):
                        idx2 = range(sum(blocksizes[:s2]), sum(blocksizes[:s2]) + blocksizes[s2])
                        coup_mat[idx1, idx2] = (coup.layer_mediated_coupling_block(vacuum_wavelength, particle1,
                                                                                   particle2, layer_system, k_parallel)
                                                + coup.direct_coupling_block(vacuum_wavelength, particle1, particle2,
                                                                             layer_system))
            else:
                raise NotImplementedError('lookups not yet implemtned')
            self.linear_operator = scipy.sparse.linalg.aslinearoperator(coup_mat)
        else:
            def evaluate_coupling_matrix(in_vec):
                out_vec = np.zeros(shape=in_vec.shape, dtype=complex)
                for n1 in range(self.n_max):
                    i1 = self.particle_index_list[n1]
                    idx1 = self.global_index_list[n1]
                    m1 = self.m_list[n1]
                    for n2 in range(self.n_max):
                        i2 = self.particle_index_list[n2]
                        idx2 = self.global_index_list[n2]
                        m2 = self.m_list[n2]
                        rho = self.particle_rho_array[i1[:, None], i2[None, :]]
                        phi = self.particle_phi_array[i1[:, None], i2[None, :]]
                        M = self.lookup[n1][n2](rho) * np.exp(1j * (m2 - m1) * phi)
                        out_vec[idx1] += M.dot(in_vec[idx2])
                return out_vec
            self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=(sum(blocksizes), sum(blocksizes)),
                                                                      matvec=evaluate_coupling_matrix,
                                                                      dtype=complex)


class SystemTMatrix:
    """Collect the particle T-matrices in a global lienear operator.

    Args:
        particle_list (list):   List of smuthi.particles.Particle objects containing a t_matrix attribute.
    """
    def __init__(self, particle_list):
        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in particle_list]
        def apply_t_matrix(vector):
            tv = np.zeros(vector.shape, dtype=complex)
            for i_s, particle in enumerate(particle_list):
                idx_block = range(sum(blocksizes[:i_s]), sum(blocksizes[:(i_s + 1)]))
                tv[idx_block] = particle.t_matrix.dot(vector[idx_block])
            return tv
        self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=(sum(blocksizes), sum(blocksizes)), 
                                                                  matvec=apply_t_matrix, matmat=apply_t_matrix,
                                                                  dtype=complex)

        
class MasterMatrix:
    r"""Represent the master matrix :math:`M = 1 - TW` as a linear operator.

    Args:
        system_t_matrix (SystemTMatrix):    T-matrix object
        coupling_matrix (CouplingMatrix):   Coupling matrix object
    """
    def __init__(self, system_t_matrix, coupling_matrix):
        if type(coupling_matrix.linear_operator).__name__ == 'MatrixLinearOperator':
            M = (np.eye(system_t_matrix.linear_operator.shape[0], dtype=complex) 
                 - system_t_matrix.linear_operator.matmat(coupling_matrix.linear_operator.A))
            self.linear_operator = scipy.sparse.linalg.aslinearoperator(M)
        else:
            def apply_master_matrix(vector):
                return vector - system_t_matrix.linear_operator.dot(coupling_matrix.linear_operator.matvec(vector))                
            self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=system_t_matrix.linear_operator.shape,
                                                                      matvec=apply_master_matrix, dtype=complex)
        