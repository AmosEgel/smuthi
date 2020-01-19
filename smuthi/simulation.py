# -*- coding: utf-8 -*-
"""Provide class to manage a simulation."""

import smuthi.linearsystem.linear_system as lsys
import smuthi.fields.coordinates_and_contours as coord
#import smuthi.utility.automatic_parameter_selection as autoparam
import sys
import os
import matplotlib.pyplot as plt
import pkg_resources
import datetime
import time
import shutil
import pickle
import numpy as np


class Simulation:
    """Central class to manage a simulation.

    Args:
        layer_system (smuthi.layers.LayerSystem):               stratified medium
        particle_list (list):                                   list of smuthi.particles.Particle objects
        initial_field (smuthi.initial_field.InitialField):      initial field object
        post_processing (smuthi.post_processing.PostProcessing): object managing post processing tasks
        k_parallel (numpy.ndarray or str):      in-plane wavenumber for Sommerfeld integrals. if 'default', keep what is
                                                in smuthi.coordinates.default_k_parallel
        solver_type (str):                      What solver type to use? 
                                                Options: 'LU' for LU factorization, 'gmres' for GMRES iterative solver
        coupling_matrix_lookup_resolution (float or None): If type float, compute particle coupling by interpolation of
                                                           a lookup table with that spacial resolution. If None
                                                           (default), don't use a lookup table but compute the coupling
                                                           directly. This is more suitable for a small particle number.
        coupling_matrix_interpolator_kind (str):  Set to 'linear' (default) or 'cubic' interpolation of the lookup table.
        store_coupling_matrix (bool):           If True (default), the coupling matrix is stored. Otherwise it is
                                                recomputed on the fly during each iteration of the solver.
        length_unit (str):      what is the physical length unit? has no influence on the computations
        input_file (str):       path and filename of input file (for logging purposes)
        output_dir (str):       path to folder where to export data
        save_after_run(bool):   if true, the simulation object is exported to disc when over
        log_to_file(bool):      if true, the simulation log will be written to a log file
        log_to_terminal(bool):  if true, the simulation progress will be displayed in the terminal
    """
    def __init__(self,
                 layer_system=None,
                 particle_list=None,
                 initial_field=None,
                 post_processing=None,
                 k_parallel='default',
                 solver_type='LU',
                 solver_tolerance=1e-4,
                 store_coupling_matrix=True,
                 coupling_matrix_lookup_resolution=None,
                 coupling_matrix_interpolator_kind='linear',
                 length_unit='length unit',
                 input_file=None,
                 output_dir='smuthi_output',
                 save_after_run=False,
                 log_to_file=False,
                 log_to_terminal=True):

        # initialize attributes
        self.layer_system = layer_system
        self.particle_list = particle_list
        self.initial_field = initial_field
        self.k_parallel = k_parallel
        self.solver_type = solver_type
        self.solver_tolerance = solver_tolerance
        self.store_coupling_matrix = store_coupling_matrix
        self.coupling_matrix_lookup_resolution = coupling_matrix_lookup_resolution
        self.coupling_matrix_interpolator_kind = coupling_matrix_interpolator_kind
        self.post_processing = post_processing
        self.length_unit = length_unit
        self.save_after_run = save_after_run

        # output
        timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        self.output_dir = output_dir + '/' + timestamp
        self.log_to_terminal = log_to_terminal
        self.log_to_file = log_to_file
        self.log_filename = self.output_dir + '/' + 'smuthi.log'
        self.set_logging()

        if input_file is not None and log_to_file:
            shutil.copyfile(input_file, self.output_dir + '/input.dat')

    def set_logging(self, log_to_terminal=None, log_to_file=None, log_filename=None):
        """Update logging behavior.

        Args:
            log_to_terminal (logical):  If true, print output to console.
            log_to_file (logical):      If true, print output to file
            log_filename (char):        If `log_to_file` is true, print output to a file with that name in the output
                                        directory. If the file already exists, it will be appended.
        """
        if log_to_terminal is not None:
            self.log_to_terminal = log_to_terminal
        if log_to_file is not None:
            self.log_to_file = log_to_file
        if log_filename is not None:
            self.log_filename = log_filename

        if not os.path.exists(self.output_dir) and self.log_to_file:
            os.makedirs(self.output_dir)
        sys.stdout = Logger(log_filename=self.log_filename,
                            log_to_file=self.log_to_file,
                            log_to_terminal=self.log_to_terminal)

    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.layer_system, self.particle_list, self.initial_field, self.k_parallel, self.solver_type,
                self.solver_tolerance, self.store_coupling_matrix, self.coupling_matrix_lookup_resolution, 
                self.coupling_matrix_interpolator_kind, self.post_processing, self.length_unit, self.save_after_run,
                coord.default_k_parallel, coord.default_polar_angles, coord.default_azimuthal_angles)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        (self.layer_system, self.particle_list, self.initial_field, self.k_parallel, self.solver_type,
         self.solver_tolerance, self.store_coupling_matrix, self.coupling_matrix_lookup_resolution,
         self.coupling_matrix_interpolator_kind, self.post_processing, self.length_unit, self.save_after_run,
         coord.default_k_parallel, coord.default_polar_angles, coord.default_azimuthal_angles) = state
        
    def print_simulation_header(self):
        sys.stdout.write("Starting simulation.\n")
        sys.stdout.flush()

    def save(self, filename=None):
        """Export simulation object to disc.

        Args:
            filename (str): path and file name where to store data
        """
        if filename is None:
            filename = self.output_dir + '/simulation.p'
        with open(filename, 'wb') as fn:
            pickle.dump(self, fn, -1)

    def initialize_linear_system(self):
        self.linear_system = lsys.LinearSystem(particle_list=self.particle_list,
                                               initial_field=self.initial_field,
                                               layer_system=self.layer_system, 
                                               k_parallel=self.k_parallel,
                                               solver_type=self.solver_type, 
                                               solver_tolerance=self.solver_tolerance,
                                               store_coupling_matrix=self.store_coupling_matrix,
                                               coupling_matrix_lookup_resolution=self.coupling_matrix_lookup_resolution,
                                               interpolator_kind=self.coupling_matrix_interpolator_kind)
    
    def run(self):
        """Start the simulation."""
        self.print_simulation_header()

        #if type(self.k_parallel) == str and self.k_parallel == "default":
        #    self.k_parallel = autoparam.default_sommerfeld_contour(self)

        # check if default contour exists, otherwise set a contour
        if coord.default_k_parallel is None:
            neff_resolution = 5e-3
            neff_max = max(np.array(self.layer_system.refractive_indices).real) + 1
            neff_imag = 1e-2
            coord.set_default_k_parallel(vacuum_wavelength=self.initial_field.vacuum_wavelength, 
                                         neff_resolution=neff_resolution, 
                                         neff_max=neff_max, 
                                         neff_imag=neff_imag)

        # prepare and solve linear system
        start = time.time()
        self.initialize_linear_system()
        self.linear_system.prepare()
        end = time.time()
        preparation_time = end - start

        start = time.time()
        self.linear_system.solve()
        end = time.time()
        solution_time = end - start

        # post processing
        start = time.time()
        if self.post_processing:
            self.post_processing.run(self)
        end = time.time()
        postprocessing_time = end - start

        if self.save_after_run:
            self.save(self.output_dir + '/simulation.p')
        
        sys.stdout.write('\n')
        sys.stdout.flush()
            
        #plt.show()

        return preparation_time, solution_time, postprocessing_time


class Logger(object):
    """Allows to prompt messages both to terminal and to log file simultaneously."""
    def __init__(self, log_filename, log_to_file=True, log_to_terminal=True, terminal=None):
        if terminal is None:
            self.terminal = sys.__stdout__

        if not log_to_terminal:
            f = open(os.devnull, 'w')
            self.terminal = f
        else:
            self.terminal = sys.__stdout__
        self.log_to_file = log_to_file
        if log_to_file:
            self.log = open(log_filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_to_file:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
