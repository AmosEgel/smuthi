# -*- coding: utf-8 -*-
import numpy as np
import smuthi.utility.optical_constants as smoc
import os

from_wl = 500 # nm
to_wl = 800 # nm
total_points = 11;
wavelengths = np.linspace(from_wl, to_wl, total_points)
single_wl = 600 # nm
test_data_path = os.path.dirname(__file__) + "/"
test_data_files = ["Si-Green-1995-test.yml",  # Should rise an error, data
                                              # type "tabulated n" is not
                                              # implemented
                   "Au-Johnson-test.yml"]


def test_yaml_read():
    for filename in test_data_files:
        try:
            index_list = smoc.read_refractive_index_from_yaml(
                test_data_path + filename, wavelengths, "nm")
            assert len(index_list) == total_points
            assert len(index_list[0]) == 2
            index_value = smoc.read_refractive_index_from_yaml(
                test_data_path + filename, single_wl, "nm")
            assert len(index_value) == 2
        except:
            if filename == "Si-Green-1995-test.yml":
                continue
            raise


if __name__ == '__main__':
    test_yaml_read()
