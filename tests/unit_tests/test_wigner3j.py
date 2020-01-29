import unittest
import numpy as np
import sympy.physics.wigner
from numba import cffi_support
from pywigxjpf_ffi import ffi, lib
import pywigxjpf_ffi

cffi_support.register_module(pywigxjpf_ffi)
pywigxjpf_ffi.lib.wig3jj

lib.wig_table_init(100, 9)
lib.wig_temp_init(100)


class TestWigner3j(unittest.TestCase):
    def test_wigxjpf_equals_numpy(self):
        """Are the results between sympy and pywigxjpf equivalent?"""
        lmax = 24
        stepsize = 8
        for m1 in range(-lmax, lmax + 1, stepsize):
            for m2 in range(-lmax, lmax + 1, stepsize):
                for l1 in range(max(1, abs(m1)), lmax + 1, stepsize):
                    for l2 in range(max(1, abs(m2)), lmax + 1, stepsize):
                        for ld in range(max(abs(l1 - l2), abs(m1 - m2)), l1 + l2 + 1, stepsize):
                            wig_sympy = float(sympy.physics.wigner.wigner_3j(l1, l2, ld, m1, -m2, -(m1 - m2)))
                            wig_pywigxjpf = pywigxjpf_ffi.lib.wig3jj(2 * l1, 2 * l2, 2 * ld, 2 * m1, -m2 * 2, -(m1 - m2) * 2)
                            print("-------------------------------------------")
                            print("l1 ", l1, "m1 ", m1, "l2", l2, "m2", m2, "ld", ld)
                            print("l1 ", l1, "m1 ", m1, "l2", l2, "m2", m2, "ld", ld)
                            print("sympy wigner3j:    ", wig_sympy)
                            print("pywigxjpf wigner3j:", wig_pywigxjpf)
                            if wig_pywigxjpf != 0:
                                print("relative difference: ", abs(wig_sympy-wig_pywigxjpf)/abs(wig_pywigxjpf))
                            np.testing.assert_almost_equal(wig_sympy, wig_pywigxjpf)


if __name__ == '__main__':
    unittest.main()