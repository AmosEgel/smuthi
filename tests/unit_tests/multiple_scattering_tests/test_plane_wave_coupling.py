# -*- coding: utf-8 -*-
import numpy as np
import smuthi.particles as part
import smuthi.particle_coupling as pacou
import smuthi.layers as lay
import smuthi.coordinates as coord
import matplotlib.pyplot as plt

wl = 550
lay_sys = lay.LayerSystem([0, 800, 0], [1, 1, 1])

# ToDo: test fails for non-coaxial particles
#spheroid1 = part.Spheroid(position=[0, 0, 400], euler_angles=[0.324, 0.567, 1.234],
#                          refractive_index=2.4 + 0.0j, semi_axis_c=50, semi_axis_a=100, l_max=2, m_max=1)
  
#spheroid2 = part.Spheroid(position=[162, 261, 253], euler_angles=[0.45, 1.23, 0.788],
#                          refractive_index=2.4 + 0.0j, semi_axis_c=100, semi_axis_a=50, l_max=2, m_max=2)

spheroid1 = part.Spheroid(position=[0, 0, 400],euler_angles=[0, 0, 0],
                          refractive_index=2.4 + 0.0j, semi_axis_c=50, semi_axis_a=100, l_max=2, m_max=1)
   
spheroid2 = part.Spheroid(position=[0,0,0],euler_angles=[0, 0, 0],
                          refractive_index=2.4 + 0.0j, semi_axis_c=100, semi_axis_a=50, l_max=2, m_max=2)

# conventional coupling using svwf addition theorem
W = pacou.direct_coupling_block(vacuum_wavelength=wl, receiving_particle=spheroid2, emitting_particle=spheroid1,
                                layer_system=lay_sys)

# plane wave coupling
k_parallel = coord.complex_contour(vacuum_wavelength=wl, neff_waypoints=[0, 0.9, 0.9-0.1j, 1.1-0.1j, 1.1, 7], 
                                   neff_resolution=1e-3)

# W_pvwf = pacou.direct_coupling_block_pvwf(vacuum_wavelength=wl, receiving_particle=spheroid2, 
#                                           emitting_particle=spheroid1, layer_system=lay_sys, k_parallel=k_parallel)

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
    err =  np.linalg.norm(W-W_pvwf) / np.linalg.norm(W) 
    print('error direct_coupling_block:', err)
    assert err < 5e-4

if __name__ == '__main__':
    test_W_block_pvwf_coupling()
              
