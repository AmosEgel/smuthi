# -*- coding: utf-8 -*-
"""Provide class to manage a simulation."""

import smuthi.linear_system as lin
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.t_matrix as tmt
import smuthi.particle_coupling as coup
import smuthi.coordinates as coord
import smuthi.post_processing as pp
import smuthi.field_expansion as fldex
import sys
import os
import matplotlib.pyplot as plt
import pkg_resources
import datetime
import shutil
import pickle
import numpy as np
import scipy.linalg



class Simulation:
    def __init__(self, layer_system=None, particle_collection=None, initial_field_collection=None, linear_system=None,
                 wr_neff_contour=None, post_processing=None, tmatrix_method=None, length_unit='length unit',
                 input_file=None, output_dir='smuthi_output', save_after_run=False):

        if layer_system is None:
            layer_system = lay.LayerSystem()
        if particle_collection is None:
            particle_collection = part.ParticleCollection()
        if initial_field_collection is None:
            initial_field_collection = init.InitialFieldCollection()
        if linear_system is None:
            linear_system = lin.LinearSystem()
        if wr_neff_contour is None:
            wr_neff_contour = coord.ComplexContour()
        if post_processing is None:
            post_processing = pp.PostProcessing()
        if tmatrix_method is None:
            tmatrix_method = {}

        self.layer_system = layer_system
        self.particle_collection = particle_collection
        self.initial_field_collection = initial_field_collection
        self.linear_system = linear_system
        self.wr_neff_contour = wr_neff_contour
        self.post_processing = post_processing
        self.tmatrix_method = tmatrix_method
        self.length_unit = length_unit
        self.save_after_run = save_after_run

        timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        self.output_dir = output_dir + '/' + timestamp
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        sys.stdout = Logger(self.output_dir + '/smuthi.log')
        if input_file is not None:
            shutil.copyfile(input_file, self.output_dir + '/input.dat')

        else:
            self.output_dir = False

    def save(self, filename=None):
        if filename is None:
            if self.output_dir:
                filename = self.output_dir + '/simulation.p'
            else:
                filename = 'simulation.p'
        with open(filename, 'wb') as fn:
            pickle.dump(self, fn, -1)

    def run(self):
        print(welcome_message())

        # compute initial field coefficients
        sys.stdout.write("Compute initial field coefficients ... ")
        sys.stdout.flush()
        self.linear_system.initial_field_coefficients = \
            init.initial_field_swe_coefficients(self.initial_field_collection, self.particle_collection,
                                                self.layer_system)
        sys.stdout.write("done. \n")

        # compute T-matrix
        sys.stdout.write("Compute T-matrices ... ")
        sys.stdout.flush()
        self.linear_system.t_matrices = \
            tmt.t_matrix_collection(self.initial_field_collection.vacuum_wavelength, self.particle_collection,
                                    self.layer_system, self.tmatrix_method)
        sys.stdout.write("done. \n")

        # compute particle coupling matrices
        sys.stdout.write("Compute direct particle coupling matrix ... ")
        sys.stdout.flush()
        self.linear_system.coupling_matrix = coup.direct_coupling_matrix(
            self.initial_field_collection.vacuum_wavelength, self.particle_collection, self.layer_system)
        sys.stdout.write("done. \n")

        sys.stdout.write("Compute layer system mediated particle coupling matrix ... ")
        sys.stdout.flush()
        self.linear_system.coupling_matrix += coup.layer_mediated_coupling_matrix(
            self.initial_field_collection.vacuum_wavelength, self.particle_collection, self.layer_system,
            self.wr_neff_contour)
        sys.stdout.write("done. \n")

        # solve linear system
        sys.stdout.write("Solve linear system ... ")
        sys.stdout.flush()
        self.linear_system.solve()
        sys.stdout.write("done. \n")

        # post processing
        sys.stdout.write("Post processing ... ")
        sys.stdout.flush()
        self.post_processing.run(self)
        sys.stdout.write("done. \n")

        if self.save_after_run:
            self.save(self.output_dir + '/simulation.p')

        plt.show()


class Logger(object):
    def __init__(self, log_filename):
        self.terminal = sys.stdout
        self.log = open(log_filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def welcome_message():
    version = pkg_resources.get_distribution("smuthi").version
    msg = ("\n"
           "********************************\n"
           "    SMUTHI version " + version + "\n"
           "********************************\n")
    return msg


class LinearSystem:
    """Linear equation for the scattered field swe coefficients."""
    def __init__(self, vacuum_wavelength, particle_collection, linear_system):

        self.scattered_field_expansion = fldex.SphericalWaveExpansion(particle_collection)
        self.initial_field_expansion = fldex.SphericalWaveExpansion(particle_collection)

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
