# -*- coding: utf-8 -*-
"""Provide class and routines for the solution of the master equation."""

import numpy as np
import scipy.linalg


class LinearSystem:
    """Linear equation for the scattered field swe coefficients."""
    def __init__(self):

        # numpy.ndarray of dimension (NS, nmax)
        self.scattered_field_coefficients = None

        # numpy.ndarray of dimension (NS, nmax)
        self.initial_field_coefficients = None

        # numpy.ndarray of dimension (NS, nmax, nmax)
        # indices are: particle number, outgoing swe index, regular swe index
        self.t_matrices = None

        # numpy.ndarray of dimension (NS, nmax, NS, nmax)
        # indices are: receiving particle number, regular swe index, emitting particle number, outgoing swe index
        self.coupling_matrix = None

        # how to solve?
        self.solver = 'LU'

        # only relevant if solver='LU'
        self.LU_piv = None

    def solve(self):
        """Compute scattered field coefficients"""
        if self.solver == 'LU':
            if self.LU_piv is None:
                lu, piv = scipy.linalg.lu_factor(self.master_matrix(), overwrite_a=False)
                self.LU_piv = (lu, piv)
            bvec = scipy.linalg.lu_solve(self.LU_piv, self.right_hand_side())
            self.scattered_field_coefficients = bvec.reshape(self.initial_field_coefficients.shape)
        else:
            raise ValueError('This solver type is currently not implemented.')

    def right_hand_side(self):
        tai = self.t_matrices * self.initial_field_coefficients[:, np.newaxis, :]
        tai = tai.sum(axis=2)
        return np.concatenate(tai)

    def master_matrix(self):
        NS = len(self.t_matrices[:, 0, 0])
        blocksize = len(self.t_matrices[0, 0, :])
        mm = np.eye(NS * blocksize, dtype=complex)
        w = np.reshape(self.coupling_matrix, (NS * blocksize, NS * blocksize))
        for s in range(NS):
            t = self.t_matrices[s, :, :]
            wblockrow = w[s * blocksize:((s + 1) * blocksize), :]
            twbl = np.dot(t, wblockrow)
            mm[s * blocksize:((s + 1) * blocksize), :] -= twbl
        return mm
