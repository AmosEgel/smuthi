# -*- coding: utf-8 -*-
"""Routines for multiple scattering. The first half of the module contains functions to explicitly compute the 
coupling matrix entries. The second half of the module contains functions for the preparation of lookup tables that 
are used to approximate the coupling matrices by interoplation."""

from numba import complex128,int64,jit
from scipy.signal.filter_design import bessel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.special
import smuthi.coordinates as coord
import smuthi.cuda_sources as cu
import smuthi.field_expansion as fldex
import smuthi.layers as lay
import smuthi.spherical_functions as sf
import smuthi.vector_wave_functions as vwf
import sys
try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule
    import pycuda.cumath
except:
    pass

