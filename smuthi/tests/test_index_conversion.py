# -*- coding: utf-8 -*-
"""Test the index_conversion module"""

import unittest
import smuthi.index_conversion


class IndexConversionTest(unittest.TestCase):
    def test_multi2single_stlm(self):
        idcs = []
        lmax = 5
        count = 0
        for s in range(3):
            for tau in range(2):
                for l in range(1, lmax + 1):
                    for m in range(-l, l + 1):
                        idcs.append(smuthi.index_conversion.multi2single(particle_number=s, tau=tau, l=l, m=m,
                                                                         lmax=lmax))
                        if s == 0:
                            count += 1
        self.assertEqual(idcs, list(range(len(idcs))))
        nmax = smuthi.index_conversion.max_index(lmax)
        self.assertEqual(count, nmax + 1)

        idcs = []
        lmax = 6
        mmax = 3
        count = 0
        for s in range(3):
            for tau in range(2):
                for l in range(1, lmax + 1):
                    mlim = min(l, mmax)
                    for m in range(-mlim, mlim + 1):
                        idcs.append(smuthi.index_conversion.multi2single(particle_number=s, tau=tau, l=l, m=m,
                                                                         lmax=lmax, mmax=mmax))
                        if s == 0:
                            count += 1
        self.assertEqual(idcs, list(range(len(idcs))))
        nmax = smuthi.index_conversion.max_index(lmax, mmax=mmax)
        self.assertEqual(count, nmax + 1)


if __name__ == '__main__':
    unittest.main()
