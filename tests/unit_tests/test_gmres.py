# -*- coding: utf-8 -*-
import sys
import unittest
import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.fields.coordinates_and_contours as coord
import smuthi.simulation as simul


class TestGMRES(unittest.TestCase):
    def testGMRES(self):
        # Parameter input ----------------------------
        vacuum_wavelength = 550
        beam_polar_angle = np.pi * 7 / 8
        beam_azimuthal_angle = np.pi * 1 / 3
        beam_polarization = 0
        beam_amplitude = 1
        beam_neff_array = np.linspace(0, 2, 501, endpoint=False)
        beam_waist = 1000
        beam_focal_point = [200, 200, 200]
        neff_waypoints = [0, 0.5, 0.8 - 0.01j, 2 - 0.01j, 2.5, 5]
        neff_discr = 1e-2
        # --------------------------------------------

        coord.set_default_k_parallel(vacuum_wavelength, neff_waypoints, neff_discr)

        # initialize particle object
        sphere1 = part.Sphere(position=[100, 100, 150], refractive_index=2.4 + 0.0j, radius=110, l_max=4, m_max=4)
        sphere2 = part.Sphere(position=[-100, -100, 250], refractive_index=1.9 + 0.0j, radius=120, l_max=3, m_max=3)
        sphere3 = part.Sphere(position=[-200, 100, 300], refractive_index=1.7 + 0.0j, radius=90, l_max=3, m_max=3)
        particle_list = [sphere1, sphere2, sphere3]

        # initialize layer system object
        lay_sys = lay.LayerSystem([0, 400, 0], [2, 1.4, 2])

        # initialize initial field object
        init_fld = init.GaussianBeam(vacuum_wavelength=vacuum_wavelength, polar_angle=beam_polar_angle,
                                     azimuthal_angle=beam_azimuthal_angle, polarization=beam_polarization,
                                     amplitude=beam_amplitude, reference_point=beam_focal_point, beam_waist=beam_waist,
                                     k_parallel_array=beam_neff_array * coord.angular_frequency(vacuum_wavelength))

        # initialize simulation object
        simulation_lu = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld,
                                         solver_type='LU', log_to_terminal='nose2' not in sys.modules.keys())  # suppress output if called by nose
        simulation_lu.run()
        coefficients_lu = particle_list[0].scattered_field.coefficients

        simulation_gmres = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld,
                                            solver_type='gmres', solver_tolerance=1e-5,
                                            log_to_terminal=(
                                                not sys.argv[0].endswith('nose2')))  # suppress output if called by nose
        simulation_gmres.run()
        coefficients_gmres = particle_list[0].scattered_field.coefficients

        np.testing.assert_allclose(np.linalg.norm(coefficients_lu), np.linalg.norm(coefficients_gmres), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
