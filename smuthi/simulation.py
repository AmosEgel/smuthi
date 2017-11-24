# -*- coding: utf-8 -*-
"""Provide class to manage a simulation."""

import smuthi.linear_system as lsys
import sys
import os
import matplotlib.pyplot as plt
import pkg_resources
import datetime
import shutil
import pickle


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
    """

    def __init__(self, layer_system=None, particle_list=None, initial_field=None, post_processing=None,
                 k_parallel='default', solver_type='LU', solver_tolerance=1e-4, store_coupling_matrix=True,
                 coupling_matrix_lookup_resolution=None, coupling_matrix_interpolator_kind='linear',
                 length_unit='length unit', input_file=None, output_dir='smuthi_output', save_after_run=False):

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
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        sys.stdout = Logger(self.output_dir + '/smuthi.log')
        if input_file is not None:
            shutil.copyfile(input_file, self.output_dir + '/input.dat')
        else:
            self.output_dir = False
    
    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.layer_system, self.particle_list, self.initial_field, self.k_parallel, self.solver_type,
                self.solver_tolerance, self.store_coupling_matrix, self.coupling_matrix_lookup_resolution, 
                self.coupling_matrix_interpolator_kind, self.post_processing, self.length_unit, self.save_after_run)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        (self.layer_system, self.particle_list, self.initial_field, self.k_parallel, self.solver_type,
         self.solver_tolerance, self.store_coupling_matrix, self.coupling_matrix_lookup_resolution,
         self.coupling_matrix_interpolator_kind, self.post_processing, self.length_unit, self.save_after_run) = state
        
    def print_simulation_header(self):
        version = pkg_resources.get_distribution("smuthi").version
        welcome_msg = ("\n" + "*" * 32 + "\n    SMUTHI version " + version + "\n" + "*" * 32 + "\n")
        sys.stdout.write(welcome_msg)
        sys.stdout.flush()

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
        self.print_simulation_header()

        self.linear_system = lsys.LinearSystem(particle_list=self.particle_list, initial_field=self.initial_field,
                                               layer_system=self.layer_system, k_parallel=self.k_parallel,
                                               solver_type=self.solver_type, solver_tolerance=self.solver_tolerance,
                                               store_coupling_matrix=self.store_coupling_matrix,
                                               coupling_matrix_lookup_resolution=self.coupling_matrix_lookup_resolution,
                                               interpolator_kind=self.coupling_matrix_interpolator_kind)
        self.linear_system.solve()

        # post processing
        if self.post_processing:
            self.post_processing.run(self)
            
        if self.save_after_run:
            self.save(self.output_dir + '/simulation.p')
        
        sys.stdout.write('\n')
        sys.stdout.flush()
            
        plt.show()


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
