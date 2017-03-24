# -*- coding: utf-8 -*-
"""Test the functions defined in vector_wave_functions.py."""

import unittest
import numpy as np
import smuthi.initial_field as init
import smuthi.layers
import smuthi.particles
import smuthi.index_conversion as idx

ld = 550
A = 1
A2 = 2
beta = np.pi*6/7
beta2 = np.pi*5/7
alpha = np.pi/3
alpha2 =  1.2* np.pi
pol = 0
pol2 = 1
rS = [100, 200, 300]
rS2 =  [200, -200, 200]
lmax = 3
laysys = smuthi.layers.LayerSystem(thicknesses=[0, 500, 0], refractive_indices=[1, 2, 1])
laysys1 = smuthi.layers.LayerSystem(thicknesses=[0, 500, 0], refractive_indices = [1, 1, 1])
laysys2 = smuthi.layers.LayerSystem(thicknesses=[0, 0], refractive_indices=[1, 1])
index_specs = idx.swe_specifications(lmax)


class PlaneWaveTest(unittest.TestCase):
    def test_SWE_coefficients_consistency(self):

        aI1 = init.planewave_swe_coefficients(vacuum_wavelength=ld, amplitude=A, polar_angle=beta,
                                              azimuthal_angle=alpha, polarization=pol,
                                              particle_position=rS, layer_system=laysys1, index_specs= index_specs)
        aI2 = init.planewave_swe_coefficients(vacuum_wavelength=ld, amplitude=A, polar_angle=beta,
                                              azimuthal_angle=alpha, polarization=pol,
                                              particle_position=rS, layer_system=laysys2, index_specs=index_specs)

        np.testing.assert_allclose(aI1, aI2)

    def test_SWE_coefficients_against_prototype(self):
        aI = init.planewave_swe_coefficients(vacuum_wavelength=ld, amplitude=A, polar_angle=beta,
                                             azimuthal_angle=alpha, polarization=pol,
                                             planewave_reference_point=[0, 0, 500],
                                             particle_position=rS, layer_system=laysys, index_specs=index_specs)
        self.assertAlmostEqual(aI[0], 0.037915264196848 + 0.749562792043970j)
        self.assertAlmostEqual(aI[5], 0.234585233040185 - 0.458335592154664j)
        self.assertAlmostEqual(aI[10], -0.047694884547150 - 0.942900216698188j)
        self.assertAlmostEqual(aI[20], 0)
        self.assertAlmostEqual(aI[29], -0.044519302207787 - 0.073942545543654j)


class InitialFieldClassTest(unittest.TestCase):
    def test_initial_field_class(self):
        in_fld = smuthi.initial_field.InitialFieldCollection(ld)
        in_fld.add_planewave(amplitude=A, polar_angle=beta, azimuthal_angle=alpha,
                             polarization=pol)
        in_fld.add_planewave(amplitude=A2, polar_angle=beta2, azimuthal_angle=alpha2,
                             polarization=pol2)
        prtcl = smuthi.particles.ParticleCollection()
        prtcl.add_sphere(100, 3, [100, 200, 300])
        prtcl.add_sphere(100, 3, [200, -200, 200])
        aI = init.initial_field_swe_coefficients(initial_field_collection=in_fld,
                                                 particle_collection=prtcl, layer_system=laysys, index_specs=index_specs)
        aI1 = init.planewave_swe_coefficients(vacuum_wavelength=ld, amplitude=A, polar_angle=beta,
                                             azimuthal_angle=alpha, polarization=pol,
                                             particle_position=rS2, layer_system=laysys, index_specs=index_specs)
        aI2 = init.planewave_swe_coefficients(vacuum_wavelength=ld, amplitude=A2, polar_angle=beta2,
                                             azimuthal_angle=alpha2, polarization=pol2,
                                             particle_position=rS2, layer_system=laysys, index_specs=index_specs)
        self.assertAlmostEqual(aI[1, 0], aI1[0] + aI2[0])


if __name__ == '__main__':
    unittest.main()
