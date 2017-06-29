import numpy as np
import smuthi.initial_field as init
import smuthi.layers
import smuthi.particles

ld = 550
A = 1
beta = np.pi*6/7
alpha = np.pi/3
pol = 0
rS = [100, 200, 300]
rS2 = [200, -200, 200]
laysys = smuthi.layers.LayerSystem(thicknesses=[0, 500, 0], refractive_indices=[1, 2, 1])
particle = smuthi.particles.Sphere(position=rS, l_max=3, m_max=3)
particle2 = smuthi.particles.Sphere(position=rS2, l_max=3, m_max=3)
particle_collection = smuthi.particles.ParticleCollection()
particle_collection.add(particle)
particle_collection.add(particle2)
plane_wave = init.PlaneWave(vacuum_wavelength=ld, polar_angle=beta, azimuthal_angle=alpha, polarization=pol,
                            amplitude=A, reference_point=[0, 0, 500])
swe = plane_wave.spherical_wave_expansion(particle_collection, laysys)


def test_SWE_coefficients_against_prototype():
    aI = swe.coefficients
    np.testing.assert_allclose(aI[0], 0.037915264196848 + 0.749562792043970j)
    np.testing.assert_allclose(aI[0], 0.037915264196848 + 0.749562792043970j)
    np.testing.assert_allclose(aI[5], 0.234585233040185 - 0.458335592154664j)
    np.testing.assert_allclose(aI[10], -0.047694884547150 - 0.942900216698188j)
    np.testing.assert_allclose(aI[20], 0)
    np.testing.assert_allclose(aI[29], -0.044519302207787 - 0.073942545543654j)


if __name__ == '__main__':
    test_SWE_coefficients_against_prototype()
