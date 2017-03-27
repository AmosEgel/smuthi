# -*- coding: utf-8 -*-
"""Provide class and routines for the solution of the master equation."""

import numpy as np
import scipy.linalg
import smuthi.index_conversion as idx
import smuthi.coordinates as coord


class LinearSystem:
    """Linear equation for the scattered field swe coefficients."""
    def __init__(self, lmax=None, mmax=None, index_arrangement=None, swe_specs=None):

        if swe_specs is None:
            self.swe_specs = idx.swe_specifications(lmax=lmax, mmax=mmax, index_arrangement=index_arrangement)
        else:
            self.swe_specs = swe_specs

        # numpy.ndarray of dimension (NS, nmax)
        self.scattered_field_coefficients = None

        # numpy.ndarray of dimension (NS, nmax)
        self.initial_field_coefficients = None

        # numpy.ndarray of dimension (NS, nmax, nmax)
        self.t_matrices = None

        # numpy.ndarray of dimension (NS*nmax, NS*nmax)
        self.coupling_matrix = None

        # how to solve?
        self.solver = 'LU'

        # only relevant if solver='LU'
        self.LU_piv = None

    def solve(self):
        """Compute scattered field coefficients"""
        if self.solver == 'LU':
            if self.LU_piv is None:
                lu, piv = scipy.linalg.lu_factor(self.master_matrix(), overwrite_a=False, check_finite=True)
                self.LU_piv = (lu, piv)
            self.scattered_field_coefficients = scipy.linalg.lu_solve(self.LU_piv, self.right_hand_side(), trans=0,
                                                                      overwrite_b=False, check_finite=True)
        else:
            raise ValueError('This solver type is currently not implemented.')

    def right_hand_side(self):
        tai = self.t_matrices * self.initial_field_coefficients[:, np.newaxis, :]
        tai = tai.sum(axis=2)
        return np.concatenate(tai)

    def master_matrix(self):
        w = self.coupling_matrix
        NS = len(self.t_matrices[:, 0, 0])
        blocksize = len(self.t_matrices[0, 0, :])
        mm = np.eye(NS * blocksize, dtype=complex)
        for s in range(NS):
            t = self.t_matrices[s, :, :]
            wblockrow = w[s * blocksize:((s + 1) * blocksize), :]
            twbl = np.dot(t, wblockrow)
            mm[s * blocksize:((s + 1) * blocksize), :] -= twbl
        return mm
