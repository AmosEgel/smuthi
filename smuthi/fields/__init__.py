"""This subpackage contains functionality that has to do with the
representation of electromagnetic fields in spherical or plane vector wave 
functions.""" 

import numpy as np

def angular_frequency(vacuum_wavelength):
    """Angular frequency :math:`\omega = 2\pi c / \lambda`

    Args:
        vacuum_wavelength (float): Vacuum wavelength in length unit

    Returns:
        Angular frequency in the units of c=1 (time units=length units). 
        This is at the same time the vacuum wavenumber.
    """
    return 2 * np.pi / vacuum_wavelength
