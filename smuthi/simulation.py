# -*- coding: utf-8 -*-
"""Provide class to manage a simulation."""

import smuthi.linear_system as lin
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.t_matrix as tmt
import smuthi.particle_coupling as coup
import smuthi.coordinates as coord
import sys


class Simulation:
    def __init__(self, layer_system=None, particle_collection=None, initial_field_collection=None, linear_system=None,
                 wr_neff_contour=None):
        """Initialize

        input:
        """
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



        self.layer_system = layer_system
        self.particle_collection = particle_collection
        self.initial_field_collection = initial_field_collection
        self.linear_system = linear_system
        self.wr_neff_contour = wr_neff_contour

    def run(self):
        clear_console()
        print(welcome_message())
        self.status_message = welcome_message()

        # compute initial field coefficients
        sys.stdout.write("Compute initial field coefficients ... ")
        sys.stdout.flush()
        self.linear_system.initial_field_coefficients = \
            init.initial_field_swe_coefficients(self.initial_field_collection, self.particle_collection,
                                                self.layer_system, self.linear_system.swe_specs)
        sys.stdout.write("done. \n")

        # compute T-matrix
        sys.stdout.write("Compute T-matrices ... ")
        sys.stdout.flush()
        self.linear_system.t_matrices = \
            tmt.t_matrix_collection(self.initial_field_collection.vacuum_wavelength, self.particle_collection,
                                    self.layer_system, self.linear_system.swe_specs)
        sys.stdout.write("done. \n")

        # compute particle coupling matrix
        sys.stdout.write("Compute particle coupling matrix ... ")
        sys.stdout.flush()
        self.linear_system.coupling_matrix = \
            coup.layer_mediated_coupling_matrix(self.initial_field_collection.vacuum_wavelength,
                                                self.particle_collection, self.layer_system,
                                                self.linear_system.swe_specs, self.wr_neff_contour)
        sys.stdout.write("done. \n")

        # solve linear system
        sys.stdout.write("Solve linear system ... ")
        sys.stdout.flush()
        self.linear_system.solve()
        sys.stdout.write("done. \n")


def welcome_message():
    msg = ("********************************\n"
           "*            SMUTHI            *\n"
           "********************************\n")
    return msg


def clear_console():
    #print("\n"*100)
    pass