import smuthi.initial_field as init
import smuthi.particles
import smuthi.coordinates as coord

ld = 550
rD = [300, -200, 100]
D = [1e7, 2e7, 3e7]
thick = [0, 200, 200, 0]
n = [1.5, 2, 1.5, 1]
waypoints = [0, 0.8, 1-0.1j, 2-0.1j, 2.1, 4]
neff_discr = 1e-3
rS = [100, 200, 300]

ctr = coord.ComplexContour(neff_waypoints=waypoints, neff_discretization=neff_discr)
dipole = init.DipoleSource(vacuum_wavelength=ld, dipole_moment=D, position=rD, contour=ctr)
laysys = smuthi.layers.LayerSystem(thicknesses=thick, refractive_indices=n)
particle = smuthi.particles.Sphere(position=rS, l_max=3, m_max=3)

aI = dipole.spherical_wave_expansion(particle, laysys)


def test_SWE_coefficients_against_prototype():
    aI0 = -43.373420715332031 - 43.502693176269531j
    err0 = abs((aI.coefficients[0] - aI0) / aI0)

    aI10 = 12.128395080566406 + 60.459373474121094j
    err10 = abs((aI.coefficients[10] - aI10) / aI10)

    aI20 = 40.009540557861328 - 83.980834960937500j
    err20 = abs((aI.coefficients[20] - aI20) / aI20)

    aI29 = -17.489177703857422 + 6.912498474121094j
    err29 = abs((aI.coefficients[29] - aI29) / aI29)

    # print(err0, err10, err20, err29)

    assert err0 < 1e-5
    assert err10 < 1e-5
    assert err20 < 1e-5
    assert err29 < 1e-5


if __name__ == '__main__':
    test_SWE_coefficients_against_prototype()
