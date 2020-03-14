# -*- coding: utf-8 -*-
import numpy as np
import smuthi.particles as part
import smuthi.linearsystem.particlecoupling.direct_coupling as pacou
import smuthi.layers as lay
import smuthi.fields as flds

wl = 550
lay_sys = lay.LayerSystem([0, 800, 0], [1, 1, 1])

spheroid1 = part.Spheroid(position=[0, 0, 400], euler_angles=[0.324, 0.567, 1.234],
                          refractive_index=2.4 + 0.0j, semi_axis_c=50, semi_axis_a=100, l_max=2, m_max=2)
  
spheroid2 = part.Spheroid(position=[162, 261, 253], euler_angles=[0.45, 1.23, 0.788],
                          refractive_index=2.4 + 0.0j, semi_axis_c=100, semi_axis_a=50, l_max=2, m_max=2)

# Test fails for mmax != lmax, 
# Computation of matrix elements in the laboratory coordinate system of degree l
# require the knowledge of all orders m of same degree l: m = (-l, -l + 1, ..., l - 1, l) 
# see e.g.:     Mishchenko et al., Scattering, Absorption, and Emission of Light by Small Particles,
#               Cambridge University Press, 2002. Eq. (5.29) p. 120

#spheroid1 = part.Spheroid(position=[0, 0, 400],euler_angles=[0, 0, 0],
#                          refractive_index=2.4 + 0.0j, semi_axis_c=50, semi_axis_a=100, l_max=2, m_max=1)
#   
#spheroid2 = part.Spheroid(position=[0,0,0],euler_angles=[0, 0, 0],
#                          refractive_index=2.4 + 0.0j, semi_axis_c=100, semi_axis_a=50, l_max=2, m_max=2)

# conventional coupling using svwf addition theorem
W = pacou.direct_coupling_block(vacuum_wavelength=wl, receiving_particle=spheroid2, emitting_particle=spheroid1,
                                layer_system=lay_sys)

# plane wave coupling
k_parallel = flds.reasonable_Sommerfeld_kpar_contour(
    vacuum_wavelength=wl,
    neff_waypoints=[0, 0.9, 0.9-0.1j, 1.1-0.1j, 1.1, 7],
    neff_resolution=1e-3)


W_pvwf = pacou.direct_coupling_block_pvwf_mediated(vacuum_wavelength=wl, receiving_particle=spheroid2,
                                                   emitting_particle=spheroid1, layer_system=lay_sys, 
                                                   k_parallel=k_parallel)


def test_W_block_pvwf_coupling():
    print(W[0, 0])
    print(W_pvwf[0, 0])
    print(W_pvwf[0, 0] / W[0, 0])
    
    print(W[1, 1])
    print(W_pvwf[1, 1])
    print(W_pvwf[1, 1] / W[1, 1])
    
#     print(W)
#     print(W_pvwf)
#     
    err = np.linalg.norm(W-W_pvwf) / np.linalg.norm(W)
    print('error direct_coupling_block:', err)
    assert err < 5e-4


if __name__ == '__main__':
    test_W_block_pvwf_coupling()
