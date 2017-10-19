import smuthi.t_matrix as tmt
import smuthi.particle_coupling as coup
import smuthi.field_expansion as fldex
import smuthi.coordinates as coord
import numpy as np
import sys
import scipy.linalg
import scipy.sparse.linalg


class LinearSystem:
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
            number of parameters describing the scattered field (int)
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
            if self.solver_type == 'LU' and hasattr(self.master_matrix.linear_operator, 'A'):
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
    def __init__(self, vacuum_wavelength, particle_list, layer_system, k_parallel='default', store_matrix=True,
                 lookup_resolution=None):
        if store_matrix:
            if lookup_resolution is None:
                blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in particle_list]
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
            raise NotImplementedError('on-the-fly not yet implemtned')


class SystemTMatrix:
    def __init__(self, particle_list):
        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in particle_list]
        def apply_t_matrix(vector):
            tv = np.zeros(vector.shape, dtype=complex)
            for i_s, particle in enumerate(particle_list):
                idx_block = range(sum(blocksizes[:i_s]), sum(blocksizes[:(i_s + 1)]))
                tv[idx_block] = particle.t_matrix.dot(vector[idx_block])
            return tv
        self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=(sum(blocksizes), sum(blocksizes)), 
                                                                  matvec=apply_t_matrix, matmat=apply_t_matrix
                                                                  , dtype=complex)

        
class MasterMatrix:
    def __init__(self, system_t_matrix, coupling_matrix):
        if type(coupling_matrix.linear_operator).__name__ == 'MatrixLinearOperator':
            M = (np.eye(system_t_matrix.linear_operator.shape[0], dtype=complex) 
                 - system_t_matrix.linear_operator.matmat(coupling_matrix.linear_operator.A))
            self.linear_operator = scipy.sparse.linalg.aslinearoperator(M)
        else:
            def apply_master_matrix(vector):
                return vector - system_t_matrix.linear_operator.dot(coupling_matrix.linear_operator.matvec(vector))                
            self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=system_t_matrix.shape, 
                                                                      matvec=apply_master_matrix, dtype=complex)
        