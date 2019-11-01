"""This module contains a class to represent the particle T-matrices for all particles, i.e., on system level."""

import numpy as np
import scipy.special
import scipy.sparse.linalg
import smuthi.linear_system as linsys


class TMatrix(linsys.SystemMatrix):
    """Collect the particle T-matrices in a global lienear operator.

    Args:
        particle_list (list):   List of smuthi.particles.Particle objects containing a t_matrix attribute.
    """
    def __init__(self, particle_list):
        linsys.SystemMatrix.__init__(self, particle_list)

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
