# -*- coding: utf-8 -*-
import sys
import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.simulation as sim
import smuthi.fields.coordinates_and_contours as coord
import smuthi.postprocessing.internal_field as ifld


# Parameter input ----------------------------
vacuum_wavelength = 550
sphere_radius = 200
n_air = 1
n_water = 1.33
n_glass = 1.5
n_metal = 1.1 + 6.1j
lmax = 3

x = [1, 24, -18, 180]
y = [0, -24, 29, 0]
z = [1, 13, -43, 0]

# initialize particle objects
diel_sphere = part.Sphere(position=[0, 0, 0], refractive_index=n_glass, radius=sphere_radius, l_max=lmax, m_max=lmax)
metal_sphere = part.Sphere(position=[0, 0, 0], refractive_index=n_metal, radius=sphere_radius, l_max=lmax, m_max=lmax)

# initialize layer system objects
lay_sys_air = lay.LayerSystem([0, 0], [n_air, n_air])
lay_sys_water = lay.LayerSystem([0, 0], [n_water, n_water])

# initialize initial field object
init_fld = init.PlaneWave(vacuum_wavelength=vacuum_wavelength, polar_angle=0, azimuthal_angle=0,
                          polarization=0, amplitude=1, reference_point=[0, 0, 0])

# simulation of dielectric sphere
simulation_diel = sim.Simulation(layer_system=lay_sys_air, particle_list=[diel_sphere], initial_field=init_fld,
                                 log_to_terminal=(not sys.argv[0].endswith('nose2')))

simulation_diel.run()

intfld_diel = ifld.internal_field_piecewise_expansion(vacuum_wavelength=vacuum_wavelength, particle_list=[diel_sphere], layer_system=lay_sys_air)
E_diel = np.array(intfld_diel.electric_field(x, y, z))


# simulation of metallic sphere
simulation_metal = sim.Simulation(layer_system=lay_sys_water, particle_list=[metal_sphere], initial_field=init_fld,
                                  log_to_terminal=(not sys.argv[0].endswith('nose2')))

simulation_metal.run()

intfld_metal = ifld.internal_field_piecewise_expansion(vacuum_wavelength=vacuum_wavelength, particle_list=[metal_sphere], layer_system=lay_sys_water)
E_metal = np.array(intfld_metal.electric_field(x, y, z))


def test_versus_celes():
    E_metal_celes = np.transpose(np.array([[0.0000000 + 0.0000000j, 0.0000015 - 0.0000010j, 0.0000000 - 0.0000000j],
                                           [-0.0000007 + 0.0000009j, 0.0000034 + 0.0000002j, 0.0000006 - 0.0000029j],
                                           [-0.0000010 - 0.0000000j, - 0.0000052 - 0.0000343j, 0.0000142 - 0.0000098j],
                                           [0.0000000 + 0.0000000j, 0.0051606 - 0.0456204j, 0.0000000 + 0.0000000j]]))

    E_diel_celes = np.transpose(np.array([[0.0000000 + 0.0000000j, 0.7665766 + 0.8608202j, 0.0000000 + 0.0000000j],
                                          [0.0144948 - 0.0053389j, 0.5851322 + 0.9504231j, -0.1699689 - 0.1021124j],
                                          [0.0102967 - 0.0049106j, 1.1293162 + 0.2558453j, 0.1316240 + 0.0713801j],
                                          [0.0000000 + 0.0000000j, 0.8560891 + 0.1384031j, 0.0000000 + 0.0000000j]]))

    for i in range(4):
        normEceles = np.linalg.norm(E_diel_celes[:, i])
        normEsmuthi = np.linalg.norm(E_diel[:, i])
        error = np.linalg.norm(E_diel[:, i] - E_diel_celes[:, i])
        rel_error = np.abs(error / normEceles)
        print("point ", i)
        print("smuthi: norm(E) = %e, celes: norm(E) = %e"%(normEsmuthi, normEceles))
        print("absolute error: %e, relative error: %e"%(error, rel_error))
        assert(error < 1e-6)

    for i in range(4):
        normEceles = np.linalg.norm(E_metal_celes[:, i])
        normEsmuthi = np.linalg.norm(E_metal[:, i])
        error = np.linalg.norm(E_metal[:, i] - E_metal_celes[:, i])
        rel_error = np.abs(error / normEceles)
        print("point ", i)
        print("smuthi: norm(E) = %e, celes: norm(E) = %e"%(normEsmuthi, normEceles))
        print("absolute error: %e, relative error: %e"%(error, rel_error))
        assert (error < 1e-6)


    #print(E_diel)
    #print(E_diel_celes)
    #assert abs((particle_list[0].scattered_field.coefficients[21] - b21) / b21) < 1e-4
    #assert abs((particle_list[0].scattered_field.coefficients[21] - b21) / b21) < 1e-4


if __name__ == '__main__':
    test_versus_celes()
