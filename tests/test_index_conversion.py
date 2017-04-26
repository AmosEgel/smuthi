# -*- coding: utf-8 -*-
"""Test the index_conversion module"""

import smuthi.index_conversion


def test_multi2single_stlm():
    idcs = []
    lmax = 5
    smuthi.index_conversion.set_swe_specs(l_max=lmax)
    count = 0

    for tau in range(2):
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                idcs.append(smuthi.index_conversion.multi_to_single_index(tau=tau, l=l, m=m))
                count += 1

    assert idcs == list(range(len(idcs)))

    ind_num = smuthi.index_conversion.number_of_indices()
    assert count == ind_num

    idcs = []
    lmax = 6
    mmax = 3
    smuthi.index_conversion.set_swe_specs(l_max=lmax, m_max=mmax)
    count = 0
    for tau in range(2):
        for l in range(1, lmax + 1):
            mlim = min(l, mmax)
            for m in range(-mlim, mlim + 1):
                idcs.append(smuthi.index_conversion.multi_to_single_index(tau=tau, l=l, m=m))
                count += 1
    assert idcs == list(range(len(idcs)))

    ind_num = smuthi.index_conversion.number_of_indices()
    assert count == ind_num


if __name__ == '__main__':
    test_multi2single_stlm()
