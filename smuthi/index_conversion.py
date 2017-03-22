# -*- coding: utf-8 -*-
"""Functions to map the multiple coefficient indices s,p,l,m to a single index n and vice versa."""


def multi2single(tau, l, m, lmax, mmax=None, particle_number=0, index_arrangement='stlm'):
    """Return a unique single index for the totality of indices characterizing a svwf expansion coefficient.

    input:
    tau:                SVWF polarization (0=spherical TE, 1=spherical TM)
    l:                  SVWF degree (1, ..., lmax)
    m:                  SVWF order (-l,...,l)
    lmax:               truncation degree of SVWF expansion
    mmax:               (optional) truncation order of SVWF expansion, i.e., |m|<=mmax, default: mmax=lmax
    particle_number:    (optional) number of particle (0,1,2,...), default=0
    index_arrangement:  (optional) string to specify the order according to which the indices are arranged
                        Possible choices are:
                        'stlm' (default), which stands for 1. particle number, 2. tau, 3. l, 4. m
                        (Other choices are not implemented at the moment.)
    """
    if mmax is None:
        mmax = lmax

    if index_arrangement == 'stlm':
        # use:
        # \sum_{l=1}^lmax (2\min(l,mmax)+1) = \sum_{l=1}^mmax (2l+1) + \sum_{l=mmax+1}^lmax (2mmax+1)
        #                                   = 2*(1/2*mmax*(mmax+1))+mmax  +  (lmax-mmax)*(2*mmax+1)
        #                                   = mmax*(mmax+2)               +  (lmax-mmax)*(2*mmax+1)

        tau_blocksize = mmax * (mmax + 2) + (lmax - mmax) * (2 * mmax + 1)
        s_blocksize = 2 * tau_blocksize
        n = particle_number * s_blocksize
        n += tau * tau_blocksize
        if (l - 1) <= mmax:
            n += (l - 1) * (l - 1 + 2)
        else:
            n += mmax * (mmax + 2) + (l - 1 - mmax) * (2 * mmax + 1)
        n += m + min(l, mmax)
        return n


def max_index(lmax, mmax=None, number_of_particles=1, index_arrangement='stlm'):
    """Return the highest index that occurs which is the number of indices minus 1.

    input:
    lmax:                truncation degree of SVWF expansion
    mmax:                (optional) truncation order of SVWF expansion, i.e., |m|<=mmax, default: mmax=lmax
    number_of_particles: (optional) total number of particles (1,2,...), default=1
    index_arrangement:   (optional) string to specify the order according to which the indices are arranged
                         Possible choices are:
                         'stlm' (default), which stands for 1. particle number, 2. tau, 3. l, 4. m
                         (Other choices are not implemented at the moment.)
    """
    if mmax is None:
        mmax = lmax
    return multi2single(tau=1, l=lmax, m=mmax, lmax=lmax, mmax=mmax, particle_number=number_of_particles - 1,
                        index_arrangement=index_arrangement)

def block_size(lmax, mmax=None, number_of_particles=1, index_arrangement='stlm'):
    """Return the total number of indices which is the maximal index plus 1.

    input:
    lmax:                truncation degree of SVWF expansion
    mmax:                (optional) truncation order of SVWF expansion, i.e., |m|<=mmax, default: mmax=lmax
    number_of_particles: (optional) total number of particles (1,2,...), default=1
    index_arrangement:   (optional) string to specify the order according to which the indices are arranged
                         Possible choices are:
                         'stlm' (default), which stands for 1. particle number, 2. tau, 3. l, 4. m
                         (Other choices are not implemented at the moment.)
    """
    return max_index(lmax=lmax, mmax=mmax, number_of_particles=number_of_particles, index_arrangement=index_arrangement) \
           + 1
