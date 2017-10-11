# -*- coding: utf-8 -*-
"""Provide class to manage a simulation."""

import smuthi.t_matrix as tmt
import smuthi.particle_coupling as coup
import smuthi.field_expansion as fldex
import smuthi.coordinates as coord
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
    """Central class to manage a simulation.

    Args:
        layer_system (smuthi.layers.LayerSystem):               stratified medium
        particle_list (list):                                   list of smuthi.particles.Particle objects
        initial_field (smuthi.initial_field.InitialField):      initial field object
        post_processing (smuthi.post_processing.PostProcessing): object managing post processing tasks
        k_parallel (numpy.ndarray or str):    in-plane wavenumber for Sommerfeld integrals. if 'default', keep what is
                                              in smuthi.coordinates.default_k_parallel
        solver (str):           what solver type to use? currently only 'LU' possible, for LU factorization
        length_unit (str):      what is the physical length unit? has no influence on the computations
        input_file (str):       path and filename of input file (for logging purposes)
        output_dir (str):       path to folder where to export data
        save_after_run(bool):   if true, the simulation object is exported to disc when over
    """

    def __init__(self, layer_system=None, particle_list=None, initial_field=None, post_processing=None, 
                 k_parallel = 'default', solver='LU', length_unit='length unit', input_file=None, 
                 output_dir='smuthi_output', save_after_run=False):

        # initialize attributes
        self.layer_system = layer_system
        self.particle_list = particle_list
        self.initial_field = initial_field
        self.post_processing = post_processing
        self.length_unit = length_unit
        self.save_after_run = save_after_run
        if type(k_parallel) == str and k_parallel == 'default':
            k_parallel = coord.default_k_parallel
        self.k_parallel = k_parallel

        # linear system
        self.solver = solver # solve with what algorithm?
        self.LU_piv = None  # will be overwritten with data if solver=='LU'

        # output
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
        """Export simulation object to disc.

        Args:
            filename (str): path and file name where to store data
        """
        if filename is None:
            if self.output_dir:
                filename = self.output_dir + '/simulation.p'
            else:
                filename = 'simulation.p'
        with open(filename, 'wb') as fn:
            pickle.dump(self, fn, -1)

    def run(self):
        """Start the simulation."""

        print(welcome_message())
        self.prepare_linear_system()
        self.solve()

        # post processing
        if self.post_processing:
            sys.stdout.write("Post processing ... ")
            sys.stdout.flush()
            self.post_processing.run(self)
            sys.stdout.write("done. \n")

        if self.save_after_run:
            self.save(self.output_dir + '/simulation.p')

        plt.show()

    def prepare_linear_system(self):
        """Computes particle coupling matrices, T-matrices and initial field coefficients."""
        
        # compute initial field coefficients
        sys.stdout.write("Compute initial field coefficients ... ")
        sys.stdout.flush()
        for particle in self.particle_list:
            particle.initial_field = self.initial_field.spherical_wave_expansion(particle, self.layer_system)
        sys.stdout.write("done. \n")

        # compute T-matrix
        sys.stdout.write("Compute T-matrices ... ")
        sys.stdout.flush()
        for particle in self.particle_list:
            niS = self.layer_system.refractive_indices[self.layer_system.layer_number(particle.position[2])]
            particle.t_matrix = tmt.t_matrix(self.initial_field.vacuum_wavelength, niS, particle)
        sys.stdout.write("done. \n")

        # compute particle coupling matrices
        sys.stdout.write("Compute direct particle coupling matrix ... ")
        sys.stdout.flush()
        self.coupling_matrix = coup.direct_coupling_matrix(self.initial_field.vacuum_wavelength, self.particle_list,
                                                           self.layer_system)
        sys.stdout.write("done. \n")

        sys.stdout.write("Compute layer system mediated particle coupling matrix ... ")
        sys.stdout.flush()
        self.coupling_matrix += coup.layer_mediated_coupling_matrix(self.initial_field.vacuum_wavelength,
                                                                    self.particle_list, self.layer_system,
                                                                    self.k_parallel)
        sys.stdout.write("done. \n")

    def number_of_unknowns(self):
        """
        Returns:
            number of parameters describing the scattered field (int)
        """
        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in self.particle_list]
        return sum(blocksizes)

    def index_block(self, iS):
        """
        Args:
            iS (int): number of particle

        Returns:
            indices that correspond to the coefficients for that particle
        """
        blocksizes = [fldex.blocksize(particle.l_max, particle.m_max) for particle in self.particle_list]
        return range(sum(blocksizes[:iS]), sum(blocksizes[:(iS + 1)]))

    def right_hand_side(self):
        r"""The right hand side of the linear system is given by :math:`\sum_{\tau l m} T^i_{\tau l m} a^i_{\tau l m }`

        Returns:
            right hand side as a complex numpy.ndarray
        """
        tai = np.zeros(self.number_of_unknowns(), dtype=complex)
        for iS, particle in enumerate(self.particle_list):
            tai[self.index_block(iS)] = particle.t_matrix.dot(particle.initial_field.coefficients)
        return tai

    def master_matrix(self):
        r"""The master matrix of the linear system is given by :math:`M=(1-T(W+W^R))`

        Returns:
            master matrix as a complex numpy.ndarray
        """
        tw = np.copy(self.coupling_matrix)
        for iS, particle in enumerate(self.particle_list):
            idx_block = self.index_block(iS)
            tw[idx_block, :] = particle.t_matrix.dot(tw[idx_block, :])
        mm = np.eye(self.number_of_unknowns(), dtype=complex) - tw
        return mm

    def solve(self):
        """Compute scattered field coefficients and store them in the particles' spherical wave expansion objects."""
        sys.stdout.write("Solve linear system ... ")
        sys.stdout.flush()
        if len(self.particle_list) > 0:
            if self.solver == 'LU':
                if self.LU_piv is None:
                    lu, piv = scipy.linalg.lu_factor(self.master_matrix(), overwrite_a=False)
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


class Logger(object):
    """Allows to prompt messages both to terminal and to log file simultaneously."""
    def __init__(self, log_filename):
        self.terminal = sys.stdout
        self.log = open(log_filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def welcome_message():
    """
    Returns:
         string to be printed when a simulation starts
    """
    version = pkg_resources.get_distribution("smuthi").version
    msg = ("\n"
           "********************************\n"
           "    SMUTHI version " + version + "\n"
           "********************************\n")
    return msg
