# -*- coding: utf-8 -*-
"""Provide functionality to read optical constants in format provided by refractiveindex.info website"""
from scipy.interpolate import interp1d
import io
import numpy as np
import yaml


def read_refractive_index_from_yaml(filename, WLs, units="mkm", kind=1):
    """Read optical constants in format provided by refractiveindex.info website.

    Args:
            filename (str): path and file name for yaml data
                            downloaded from refractiveindex.info
            WLs (float or np.array): wavelengths where refractive
                                     index data is needed
            units (str): units for wavelength. currently, microns ('mkm' or 'um')
                         and nanometers ('nm') can be selected
            kind (int): order of interpolation
    
    Returns:
            A pair (or np.array of pairs) of wavelength and
            corresponding refractive index (complex)
    """
    if units == "nm": 
        factor = 1000
    elif units in ("mkm", "um"):
        factor = 1
    else:
        raise NotImplementedError("Converting wavelength into '"+units
                                  +"' units for refractive index data"
                                  +" was not implemented.")
    
    the_file = yaml.load(open(filename))['DATA'][0]
    data_type = the_file['type']
    if data_type != 'tabulated nk':
        raise NotImplementedError("Input data type '"+data_type
                                  +"' available in file "+filename
                                  +" was not implemented.")
    data = the_file['data'].splitlines()
    data_split = []
    for wl in data:
        data_split.append(wl.split())
    data_num = []
    for wl in data_split:
        record = []
        for val in wl:
            record.append(float(val))
        data_num.append(record)
    data_np = np.array(data_num)
    data_WL = data_np[:,0]*factor
    epsRe = data_np[:,1]
    epsIm = data_np[:,2]
    fRe = interp1d(data_WL, epsRe, kind=kind)
    fIm = interp1d(data_WL, epsIm, kind=kind)
    data_out = np.transpose(np.vstack((WLs, fRe(WLs)+fIm(WLs)*1j)))
    if len(data_out) == 1:
        return data_out[0]
    return data_out
