# -*- coding: utf-8 -*-
"""Test the LayerSystem class"""

import unittest
import smuthi.layers
import numpy as np


class LayerSystemTest(unittest.TestCase):
    def test_lower_zlimit(self):
        lsys = smuthi.layers.LayerSystem(thicknesses=[0,100,200,0], refractive_indices=[1,2+2j,3,3])
        self.assertEqual(lsys.lower_zlimit(0),-np.inf)
        self.assertEqual(lsys.lower_zlimit(1), 0)
        self.assertEqual(lsys.lower_zlimit(2), 100)
        self.assertEqual(lsys.lower_zlimit(3), 300)

    def test_upper_zlimit(self):
        lsys = smuthi.layers.LayerSystem(thicknesses=[0, 100, 200, 0], refractive_indices=[1, 2 + 2j, 3, 3])
        self.assertEqual(lsys.upper_zlimit(0), 0)
        self.assertEqual(lsys.upper_zlimit(1), 100)
        self.assertEqual(lsys.upper_zlimit(2), 300)
        self.assertEqual(lsys.upper_zlimit(3), np.inf)

    def test_reference_z(self):
        lsys = smuthi.layers.LayerSystem(thicknesses=[0, 100, 200, 0], refractive_indices=[1, 2 + 2j, 3, 3])
        self.assertEqual(lsys.reference_z(0), 0)
        self.assertEqual(lsys.reference_z(1), 0)
        self.assertEqual(lsys.reference_z(2), 100)
        self.assertEqual(lsys.reference_z(3), 300)

    def test_layer_number(self):
        lsys = smuthi.layers.LayerSystem(thicknesses=[0, 100, 200, 0], refractive_indices=[1, 2 + 2j, 3, 3])
        self.assertEqual(lsys.layer_number(-100), 0)
        self.assertEqual(lsys.layer_number(0), 1)
        self.assertEqual(lsys.layer_number(50), 1)
        self.assertEqual(lsys.layer_number(100), 2)
        self.assertEqual(lsys.layer_number(150), 2)
        self.assertEqual(lsys.layer_number(250), 2)
        self.assertEqual(lsys.layer_number(350), 3)


if __name__ == '__main__':
    unittest.main()
