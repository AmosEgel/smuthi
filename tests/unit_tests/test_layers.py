# -*- coding: utf-8 -*-
"""Test the LayerSystem class"""

import smuthi.layers
import numpy as np


def test_lower_zlimit():
    lsys = smuthi.layers.LayerSystem(thicknesses=[0,100,200,0], refractive_indices=[1,2+2j,3,3])
    assert lsys.lower_zlimit(0) == -np.inf
    assert lsys.lower_zlimit(1) == 0
    assert lsys.lower_zlimit(2) == 100
    assert lsys.lower_zlimit(3) == 300


def test_upper_zlimit():
    lsys = smuthi.layers.LayerSystem(thicknesses=[0, 100, 200, 0], refractive_indices=[1, 2 + 2j, 3, 3])
    assert lsys.upper_zlimit(0) == 0
    assert lsys.upper_zlimit(1) == 100
    assert lsys.upper_zlimit(2) == 300
    assert lsys.upper_zlimit(3) == np.inf


def test_reference_z():
    lsys = smuthi.layers.LayerSystem(thicknesses=[0, 100, 200, 0], refractive_indices=[1, 2 + 2j, 3, 3])
    assert lsys.reference_z(0) == 0
    assert lsys.reference_z(1) == 0
    assert lsys.reference_z(2) == 100
    assert lsys.reference_z(3) == 300


def test_layer_number():
    lsys = smuthi.layers.LayerSystem(thicknesses=[0, 100, 200, 0], refractive_indices=[1, 2 + 2j, 3, 3])
    assert lsys.layer_number(-100) == 0
    assert lsys.layer_number(0) == 1
    assert lsys.layer_number(50) == 1
    assert lsys.layer_number(100) == 2
    assert lsys.layer_number(150) == 2
    assert lsys.layer_number(250) == 2
    assert lsys.layer_number(350) == 3


if __name__ == '__main__':
    test_layer_number()
    test_lower_zlimit()
    test_reference_z()
    test_upper_zlimit()
