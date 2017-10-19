import smuthi.t_matrix as tmt
import smuthi.particle_coupling as coup
import smuthi.field_expansion as fldex
import smuthi.coordinates as coord
import numpy as np
import sys
import scipy.linalg


class LinearSystem:
    def __init__(self, particle_list, initial_field, layer_system, k_parallel='default', solver_type='LU', store_coupling_matrix=True,
                 coupling_matrix_lookup_resolution=None):
        self.k_parallel = k_parallel
        self.solver_type = solver_type
        self.LU_piv = None

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
        self.coupling_matrix = coup.CouplingMatrix(vacuum_wavelength=initial_field.vacuum_wavelength,
                                                   particle_list=particle_list, layer_system=layer_system,
                                                   k_parallel=self.k_parallel, store_matrix=store_coupling_matrix,
                                                   lookup_resolution=coupling_matrix_lookup_resolution)
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
            if self.solver_type == 'LU' and self.coupling_matrix.store_matrix:
                if self.LU_piv is None:
                    tw = np.copy(self.coupling_matrix.matrix)
                    for iS, particle in enumerate(self.particle_list):
                        idx_block = self.index_block(iS)
                        tw[idx_block, :] = particle.t_matrix.dot(tw[idx_block, :])
                    mm = np.eye(self.number_of_unknowns(), dtype=complex) - tw
                    lu, piv = scipy.linalg.lu_factor(mm, overwrite_a=False)
                    self.LU_piv = (lu, piv)
                b = scipy.linalg.lu_solve(self.LU_piv, self.right_hand_side())
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
