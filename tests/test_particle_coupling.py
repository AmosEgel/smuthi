# -*- coding: utf-8 -*-
"""Test the functions defined in particle_coupling.py."""

import unittest
import smuthi.particle_coupling
import smuthi.layers
import smuthi.index_conversion
import smuthi.coordinates


wl = 550
rs1 = [0, 0, 250]
rs2 = [100, -400, 150]
laysys_substrate = smuthi.layers.LayerSystem(thicknesses=[0, 0], refractive_indices=[2 + 0.1j, 1])
laysys_waveguide = smuthi.layers.LayerSystem(thicknesses=[0, 500, 0], refractive_indices=[1, 2, 1])
swe_idx_specs = smuthi.index_conversion.swe_specifications(3, 3)
neff_contour = smuthi.coordinates.ComplexContour([0, 0.8, 0.8-0.1j, 1.2-0.1j, 1.2, 5], 2e-3)


class WrTest(unittest.TestCase):
    def test_wr_against_prototype_single_particle_over_substrate(self):
        wr = smuthi.particle_coupling.layer_mediated_coupling_block(wl, rs1, rs1, laysys_substrate, swe_idx_specs,
                                                                    neff_contour, show_integrand=False)
        wr00 = -0.031367838040052 - 0.086576016508040j
        self.assertTrue((wr[0, 0] - wr00) / wr00 < 1e-4)
        wr1910 = 0.110962483421277 + 0.155167528762370j
        self.assertTrue((wr[19, 10] - wr1910) / wr1910 < 1e-4)


if __name__ == '__main__':
    unittest.main()
