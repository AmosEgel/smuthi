# -*- coding: utf-8 -*-
"""Functions to map the multiple coefficient indices tau,l,m to a single index n and vice versa."""

# the global variable index_order specifies the order of the multipole indices (tau, l, m) of the spherical wave
# expansion
# index_order = 'tlm' implies the following map between multi-index and single-index:
# n | tau, l, m
# -------------
# 1 | 1, 1, -1
# 2 | 1, 1, 0
# 3 | 1, 1, 1
# 2 | 1, 2, -2
# 2 | 1, 2, -1
# 2 | 1, 2, 0
# ..| ... ...
# ..| 1, lmax, lmax
# ..| 2, 1, -1
# ..| ... ...
index_order = 'tlm'

# global variable for maximal multipole degree
l_max = None

# global variable for maximal multipole order
m_max = None


def multi_to_single_index(tau, l, m):
    """Return a unique single index for the totality of indices characterizing a svwf expansion coefficient.

    input:
    tau:                SVWF polarization (0=spherical TE, 1=spherical TM)
    l:                  SVWF degree (1, ..., lmax)
    m:                  SVWF order (-l,...,l)
    """
    if index_order == 'tlm':
        # use:
        # \sum_{l=1}^lmax (2\min(l,mmax)+1) = \sum_{l=1}^mmax (2l+1) + \sum_{l=mmax+1}^lmax (2mmax+1)
        #                                   = 2*(1/2*mmax*(mmax+1))+mmax  +  (lmax-mmax)*(2*mmax+1)
        #                                   = mmax*(mmax+2)               +  (lmax-mmax)*(2*mmax+1)

        tau_blocksize = m_max * (m_max + 2) + (l_max - m_max) * (2 * m_max + 1)
        n = tau * tau_blocksize
        if (l - 1) <= m_max:
            n += (l - 1) * (l - 1 + 2)
        else:
            n += m_max * (m_max + 2) + (l - 1 - m_max) * (2 * m_max + 1)
        n += m + min(l, m_max)
        return n


def number_of_indices():
    """Return the total number of indices which is the maximal index plus 1."""
    return multi_to_single_index(tau=1, l=l_max, m=m_max) + 1


def set_swe_specs(**kwargs):
    """
    Set the truncation degree and order of the spherical wave expansion, as well as the index order for the single to
    multi index mapping.

    key-word input:
    l_max:              set the global truncation degree of SVWF expansions
    m_max:              set the global truncation order of SVWF expansions. None means m_max = l_max
    index_order:        set string to globally specify the order according to which the indices are arranged
                        Possible choices are:
                        'tlm' (default), which stands for 1. tau, 2. l, 3. m
                        (Other choices are not implemented at the moment.)
    """
    global l_max, m_max, index_order

    # set global variables
    if 'l_max' in kwargs:
        l_max = kwargs['l_max']
        m_max = l_max
    if 'm_max' in kwargs:
        m_max = kwargs['m_max']
    if 'index_order' in kwargs:
        index_order = kwargs['index_order']
    for key in kwargs.keys():
        if not (key == 'l_max' or key == 'm_max' or key == 'index_order'):
            raise ValueError('Unknown argument ' + key)
