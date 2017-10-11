import smuthi.initial_field as init
import smuthi.particles
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.layers as lay
import numpy as np

# Parameter input ----------------------------
ld = 550
rD = [300, -200, 100]
D = [1e7, 2e7, 3e7]
thick = [0, 200, 200, 0]
n = [1, 1.5, 2+1e-2j, 1+5j]
waypoints = [0, 0.8, 0.8-0.1j, 2.1-0.1j, 2.1, 4]
neff_discr = 1e-2
rS = [100, 200, 300]
nS = 1.5
RS = 100
# --------------------------------------------

coord.set_default_k_parallel(vacuum_wavelength=ld, neff_waypoints=waypoints, neff_resolution=neff_discr)
dipole = init.DipoleSource(vacuum_wavelength=ld, dipole_moment=D, position=rD)
laysys = lay.LayerSystem(thicknesses=thick, refractive_indices=n)
particle = smuthi.particles.Sphere(position=rS, l_max=3, m_max=3, refractive_index=nS, radius=RS)
simulation = simul.Simulation(layer_system=laysys, particle_list=[particle], initial_field=dipole)

aI = dipole.spherical_wave_expansion(particle, laysys)


def test_SWE_coefficients_against_prototype():
    aI0 = -0.042563330382109 - 1.073039889335632j
    err0 = abs((aI.coefficients[0] - aI0) / aI0)

    aI10 = 0.496704638004303 + 0.908744692802429j
    err10 = abs((aI.coefficients[10] - aI10) / aI10)

    aI20 = 1.209331750869751 - 2.165398836135864j
    err20 = abs((aI.coefficients[20] - aI20) / aI20)

    aI29 = -0.801326990127563 - 0.706078171730042j
    err29 = abs((aI.coefficients[29] - aI29) / aI29)

    print(err0, err10, err20, err29)

    assert err0 < 1e-4
    assert err10 < 1e-4
    assert err20 < 1e-4
    assert err29 < 1e-4


if __name__ == '__main__':
    test_SWE_coefficients_against_prototype()
