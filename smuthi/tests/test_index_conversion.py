# -*- coding: utf-8 -*-
"""Test the index_conversion module"""

import unittest
import smuthi.index_conversion


class IndexConversionTest(unittest.TestCase):
    def test_multi2single_stlm(self):
        idcs = []
        lmax = 5
        for s in range(3):
            for tau in range(2):
                for l in range(1, lmax + 1):
                    for m in range(-l, l + 1):
                        idcs.append(smuthi.index_conversion.multi2single(particle_number=s, tau=tau, l=l, m=m, lmax=lmax))
        self.assertEqual(idcs, list(range(len(idcs))))

        idcs = []
        lmax = 6
        mmax = 3
        for s in range(3):
            for tau in range(2):
                for l in range(1, lmax + 1):
                    mlim = min(l, mmax)
                    for m in range(-mlim, mlim + 1):
                        idcs.append(
                            smuthi.index_conversion.multi2single(particle_number=s, tau=tau, l=l, m=m, lmax=lmax, mmax=mmax))
        self.assertEqual(idcs, list(range(len(idcs))))


if __name__ == '__main__':
    unittest.main()
