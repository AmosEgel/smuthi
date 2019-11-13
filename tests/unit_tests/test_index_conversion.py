# -*- coding: utf-8 -*-
"""Test the index_conversion module"""
import unittest
import smuthi.fields.expansions as fldex


class TestIndexConversion(unittest.TestCase):
    def testMulti2SingleSTLM(self):
        idcs = []
        lmax = 5
        mmax = 5
        count = 0
        for tau in range(2):
            for l in range(1, lmax + 1):
                for m in range(-l, l + 1):
                    idcs.append(fldex.multi_to_single_index(tau=tau, l=l, m=m, l_max=lmax, m_max=mmax))
                    count += 1

        self.assertEqual(idcs, list(range(len(idcs))))

        ind_num = fldex.blocksize(lmax, mmax)
        self.assertEqual(count, ind_num)

        idcs = []
        lmax = 6
        mmax = 3
        count = 0
        for tau in range(2):
            for l in range(1, lmax + 1):
                mlim = min(l, mmax)
                for m in range(-mlim, mlim + 1):
                    idcs.append(fldex.multi_to_single_index(tau=tau, l=l, m=m, l_max=lmax, m_max=mmax))
                    count += 1
        self.assertEqual(idcs, list(range(len(idcs))))

        ind_num = fldex.blocksize(lmax, mmax)
        self.assertEqual(count, ind_num)


if __name__ == '__main__':
    unittest.main()
