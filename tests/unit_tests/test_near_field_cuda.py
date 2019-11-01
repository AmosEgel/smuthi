import sys
import smuthi.initial_field as init
import smuthi.particles as part
import smuthi.fields.coordinates_and_contours as coord
import smuthi.simulation as simul
import smuthi.layers as lay
import smuthi.postprocessing.scattered_field as sf
import smuthi.utility.cuda as cu
import numpy as np

ld = 550
rD = [100, -100, 100]
D = [1e7, 2e7, 3e7]
waypoints = [0, 0.8, 0.8-0.1j, 2.1-0.1j, 2.1, 4]
neff_discr = 2e-2

coord.set_default_k_parallel(vacuum_wavelength = ld, neff_waypoints=waypoints, neff_resolution=neff_discr)

# initialize particle object
sphere1 = part.Sphere(position=[200, 200, 300], refractive_index=2.4 + 0.0j, radius=110, l_max=3, m_max=3)
sphere2 = part.Sphere(position=[-200, -200, 300], refractive_index=2.4 + 0.0j, radius=120, l_max=3, m_max=3)
sphere3 = part.Sphere(position=[-200, 200, 300], refractive_index=2.5 + 0.0j, radius=90, l_max=3, m_max=3)
part_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [1+6j, 2.3, 1.5])

# initialize dipole object
dipole = init.DipoleSource(vacuum_wavelength=ld, dipole_moment=D, position=rD)

# run simulation
simulation = simul.Simulation(layer_system=lay_sys, particle_list=part_list, initial_field=dipole,
                              log_to_terminal=(not sys.argv[0].endswith('nose2')))  # suppress output if called by nose
simulation.run()

xarr = np.array([-300, 400, -100, 200])
yarr = np.array([200, -100, 400, 300])
zarr = np.array([-50, 200, 600, 700])

scat_fld_exp = sf.scattered_field_piecewise_expansion(ld, part_list, lay_sys)
e_x_scat_cpu, e_y_scat_cpu, e_z_scat_cpu = scat_fld_exp.electric_field(xarr, yarr, zarr) 
e_x_init_cpu, e_y_init_cpu, e_z_init_cpu = simulation.initial_field.electric_field(xarr, yarr, zarr, lay_sys)

cu.enable_gpu()
scat_fld_exp = sf.scattered_field_piecewise_expansion(ld, part_list, lay_sys)
e_x_scat_gpu, e_y_scat_gpu, e_z_scat_gpu = scat_fld_exp.electric_field(xarr, yarr, zarr) 
e_x_init_gpu, e_y_init_gpu, e_z_init_gpu = simulation.initial_field.electric_field(xarr, yarr, zarr, lay_sys)


def test_electric_field():
    err_scat_x = np.linalg.norm(e_x_scat_cpu - e_x_scat_gpu) / np.linalg.norm(e_x_scat_cpu)
    err_in_x = np.linalg.norm(e_x_init_cpu - e_x_init_gpu) / np.linalg.norm(e_x_init_cpu)
    err_scat_y = np.linalg.norm(e_y_scat_cpu - e_y_scat_gpu) / np.linalg.norm(e_y_scat_cpu)
    err_in_y = np.linalg.norm(e_y_init_cpu - e_y_init_gpu) / np.linalg.norm(e_y_init_cpu)
    err_scat_z = np.linalg.norm(e_z_scat_cpu - e_z_scat_gpu) / np.linalg.norm(e_z_scat_cpu)
    err_in_z = np.linalg.norm(e_z_init_cpu - e_z_init_gpu) / np.linalg.norm(e_z_init_cpu)
    
    print('scattered field component errors: ', err_scat_x, err_scat_y, err_scat_z)
    print('initial field component errors: ', err_in_x, err_in_y, err_in_z)
    assert err_scat_x < 1e-5
    assert err_scat_y < 1e-5
    assert err_scat_z < 1e-5
    assert err_in_x < 1e-5
    assert err_in_y < 1e-5
    assert err_in_z < 1e-5


if __name__ == '__main__':
    test_electric_field()
