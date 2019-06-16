import smuthi.initial_field as init
import smuthi.particles as part
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.layers as lay
import smuthi.scattered_field as sf
import numpy as np


ld = 550
rD1 = [100, -100, 100]
D1 = [1e7, 2e7, 3e7]
rD2 = [-100, 100, -100]
D2 = [-2e7, 3e7, 1e7]

waypoints = [0, 0.8, 0.8-0.1j, 2.1-0.1j, 2.1, 3]
neff_max = 3
neff_discr = 5e-3


#coord.set_default_k_parallel(vacuum_wavelength = ld, neff_resolution=neff_discr, neff_max=neff_max)
#coord.set_default_k_parallel(vacuum_wavelength = ld, neff_waypoints=waypoints, neff_resolution=neff_discr, neff_max=neff_max)
#coord.default_k_parallel = np.array([0, 0.5*2*np.pi/ld])

# we avoid to use default_k_parallel, because there is some issue when running this test with nose2 ...
kpar = coord.complex_contour(ld, waypoints, neff_discr)

# initialize particle object
# first two spheres in top layer
sphere1 = part.Sphere(position=[200, 200, 500], refractive_index=2.4 + 0.0j, radius=110, l_max=3, m_max=3)
sphere2 = part.Sphere(position=[200, -200, 500], refractive_index=2.4 + 0.0j, radius=110, l_max=3, m_max=3)
# third sphere is in same layer as third dipole
sphere3 = part.Sphere(position=[-200, -200, -400], refractive_index=2.5 + 0.0j, radius=90, l_max=3, m_max=3)

part_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [1, 2, 1.5])

# initialize dipole object
dipole1 = init.DipoleSource(vacuum_wavelength=ld, dipole_moment=D1, position=rD1, k_parallel=kpar)
dipole2 = init.DipoleSource(vacuum_wavelength=ld, dipole_moment=D2, position=rD2, k_parallel=kpar)

dipole_collection = init.DipoleCollection(vacuum_wavelength=ld, k_parallel_array=kpar)
dipole_collection.append(dipole1)
dipole_collection.append(dipole2)

dipole_collection_pwe = init.DipoleCollection(vacuum_wavelength=ld, k_parallel_array=kpar, compute_swe_by_pwe=True)
dipole_collection_pwe.append(dipole1)
dipole_collection_pwe.append(dipole2)

def test_swe_methods_agree():
    swe1 = dipole_collection.spherical_wave_expansion(sphere1, lay_sys)
    swe1_pwe = dipole_collection_pwe.spherical_wave_expansion(sphere1, lay_sys)
    err = np.linalg.norm(swe1.coefficients - swe1_pwe.coefficients)
    print('error sphere 1', err)
    assert err < 1e-4

    swe2 = dipole_collection.spherical_wave_expansion(sphere2, lay_sys)
    swe2_pwe = dipole_collection_pwe.spherical_wave_expansion(sphere2, lay_sys)
    err = np.linalg.norm(swe2.coefficients - swe2_pwe.coefficients)
    print('error sphere 2', err)
    assert err < 1e-4

    swe3 = dipole_collection.spherical_wave_expansion(sphere3, lay_sys)
    swe3_pwe = dipole_collection_pwe.spherical_wave_expansion(sphere3, lay_sys)
    err = np.linalg.norm(swe3.coefficients - swe3_pwe.coefficients)
    print('error sphere 3', err)
    assert err < 1e-4


if __name__ == '__main__':
    test_swe_methods_agree()
