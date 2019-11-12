# -*- coding: utf-8 -*-
import unittest
import functools

import numpy as np
import smuthi.utility.optical_constants as smoc
import os

from_wl = 500  # nm
to_wl = 800  # nm
total_points = 11
wavelengths = np.linspace(from_wl, to_wl, total_points)
single_wl = 600  # nm
test_data_path = os.path.dirname(__file__) + "/"


class TestYamlRead(unittest.TestCase):
    def testException(self):
        # Should rise an error, data type "tabulated n" is not implemented
        filename = "Si-Green-1995-test.yml"
        my_callable = functools.partial(smoc.read_refractive_index_from_yaml, test_data_path + filename, wavelengths, "nm")
        self.assertRaises(NotImplementedError, my_callable)

    def testYamlRead(self):
        filename = "Au-Johnson-test.yml"
        index_list = smoc.read_refractive_index_from_yaml(
            test_data_path + filename, wavelengths, "nm")
        self.assertEqual(len(index_list), total_points)
        self.assertEqual(len(index_list[0]), 2)

        index_value = smoc.read_refractive_index_from_yaml(
            test_data_path + filename, single_wl, "nm")

        self.assertEqual(len(index_value), 2)


if __name__ == '__main__':
    unittest.main()
