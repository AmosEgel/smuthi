"""This subpackage contains functionality that has to do with the
representation of electromagnetic fields in spherical or plane vector wave 
functions.""" 

import numpy as np
import sys
try:
    from numba import cffi_support
    from pywigxjpf_ffi import ffi, lib
    import pywigxjpf_ffi
    cffi_support.register_module(pywigxjpf_ffi)
    nb_wig3jj = pywigxjpf_ffi.lib.wig3jj

    lib.wig_table_init(100,9)
    lib.wig_temp_init(100)
except:
    sys.stdout.write('No pywigxjpf installation found, '
    'using sympy implementaion of Wigner-3j symbols instead.\n'
    'In certain cases, this can significantly increase the simulation time.\n'
    'You can try "pip install pywigxjpf".\n')
    sys.stdout.flush()
    from sympy.physics.wigner import wigner_3j
    def nb_wig3jj(jj_1, jj_2, jj_3, mm_1, mm_2, mm_3):
        return float(wigner_3j(jj_1/2, jj_2/2, jj_3/2, mm_1/2, mm_2/2, mm_3/2))


def angular_frequency(vacuum_wavelength):
    """Angular frequency :math:`\omega = 2\pi c / \lambda`

    Args:
        vacuum_wavelength (float): Vacuum wavelength in length unit

    Returns:
        Angular frequency in the units of c=1 (time units=length units). 
        This is at the same time the vacuum wavenumber.
    """
    return 2 * np.pi / vacuum_wavelength
