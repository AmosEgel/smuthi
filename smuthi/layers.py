# -*- coding: utf-8 -*-
"""Provide class for the representation of planar layer systems."""

import numpy as np


class LayerSystem:
    """Stack of planar layers."""

    def __init__(self, thicknesses=[0, 0], refractive_indices=[1, 1]):
        """Initialize

        input:
        thicknesses         list of layer thicknesses, first and last are semi inf and set to 0 (length unit)
        refractive_indices  list of complex refractive indices in the form n+jk
        """
        self.thicknesses = thicknesses
        self.thicknesses[0] = 0
        self.thicknesses[-1] = 0
        self.refractive_indices = refractive_indices

    def number_of_layers(self):
        """Return total number of layers"""
        return len(self.thicknesses)

    def lower_zlimit(self, i):
        """Return the z-coordinate of lower boundary

        The coordinate system is defined such that z=0 corresponds to the interface between layer 0 and layer 1.

        input:
        i:      index of layer in question (must be between 0 and number_of_layers-1)
        """
        if i == 0:
            return -np.inf
        else:
            sumthick = 0
            for d in self.thicknesses[1:i]:
                sumthick += d
        return sumthick

    def upper_zlimit(self, i):
        """Return the z-coordinate of upper boundary.

        The coordinate system is defined such that z=0 corresponds to the interface between layer 0 and layer 1.

        input:
        i:      index of layer in question (must be between 0 and number_of_layers-1)
        """
        if i == self.number_of_layers() - 1:
            return np.inf
        else:
            sumthick = 0
            for d in self.thicknesses[1:i + 1]:
                sumthick += d
        return sumthick

    def reference_z(self, i):
        """Return the anchor point's z-coordinate.

        The coordinate system is defined such that z=0 corresponds to the interface between layer 0 and layer 1.

        input:
        i:      index of layer in question (must be between 0 and number_of_layers-1)
        """
        if i == 0:
            return self.upper_zlimit(i)
        else:
            return self.lower_zlimit(i)

    def layer_number(self, z):
        """ Return number of layer that contains point [0,0,z]

        If z is on the interface, the higher layer number is selected.

        input:
        z:       z-coordinate of query point (length unit)
        """
        d = 0
        laynum = 0
        for th in self.thicknesses[1:]:
            if z >= d:
                laynum += 1
                d += th
            else:
                break
        return laynum
