#*****************************************************************************#
# This is a simple example script for Smuthi v1.0.0.                          #
# It evaluates the differential scattering cross section of a large number of #
# glass spheres on a glass substrate, excited by a plane wave under normal    #
# incidence.                                                                  #
#*****************************************************************************#

import numpy as np
import matplotlib.pyplot as plt
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.far_field
import smuthi.postprocessing.graphical_output
import smuthi.utility.cuda as cu


# In this file, all lengths are given in nanometers

# particle configuration
def vogel_spiral(number_of_spheres):
    spheres_list = []
    for i in range(1, number_of_spheres):
        r = 200 * np.sqrt(i)
        theta = i * 137.508 * np.pi/180
        spheres_list.append(smuthi.particles.Sphere(position=[r*np.cos(theta),
                                                               r*np.sin(theta),
                                                               100],
                                                    refractive_index=1.52,
                                                    radius=100,
                                                    l_max=3))
    return spheres_list


# set up and run a simulation
def simulate_N_spheres(number_of_spheres=100,
                       direct_inversion=True,
                       use_gpu=False,
                       solver_tolerance=5e-4,
                       lookup_resolution=5,
                       interpolation_order='linear',
                       make_illustrations=False):

    # Initialize the layer system: substrate (glass) and ambient (air)
    two_layers = smuthi.layers.LayerSystem(thicknesses=[0, 0],
                                           refractive_indices=[1.52, 1])
    
    # Initial field
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=550,
                                                polar_angle=np.pi,  # from top
                                                azimuthal_angle=0,
                                                polarization=0)  # 0=TE 1=TM
    
    spheres_list = vogel_spiral(number_of_spheres)
    
    # Initialize and run simulation
    cu.enable_gpu(use_gpu)
    if use_gpu and not cu.use_gpu:
        print("Failed to load pycuda, skipping simulation")
        return [0, 0, 0]
   
    if direct_inversion:
        simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                                  particle_list=spheres_list,
                                                  initial_field=plane_wave)
    else:
        simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                                  particle_list=spheres_list,
                                                  initial_field=plane_wave,
                                                  solver_type='gmres',
                                                  solver_tolerance=solver_tolerance,
                                                  store_coupling_matrix=False,
                                                  coupling_matrix_lookup_resolution=lookup_resolution,
                                                  coupling_matrix_interpolator_kind=interpolation_order)

    preparation_time, solution_time, _ = simulation.run()

    # compute cross section
    ecs = smuthi.postprocessing.far_field.extinction_cross_section(
        initial_field=plane_wave,
        particle_list=spheres_list,
        layer_system=two_layers)

    if make_illustrations:
        azimuthal_angles = np.arange(0, 361, 0.5, dtype=float) * np.pi / 180
        polar_angles = np.arange(0, 181, 0.25, dtype=float) * np.pi / 180
        dscs = smuthi.postprocessing.far_field.scattering_cross_section(
            initial_field=plane_wave,
            particle_list=spheres_list,
            layer_system=two_layers,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
        )

        # display differential scattering cross section
        smuthi.postprocessing.graphical_output.show_far_field(
            far_field=dscs,
            save_plots=True,
            show_plots=True,
            save_data=False,
            tag='dscs_%ispheres'%number_of_spheres,
            log_scale=True)

        xlim = max([abs(sphere.position[0]) for sphere in spheres_list]) * 1.1
        plt.figure()
        plt.xlim([-xlim, xlim])
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.ylim([-xlim, xlim])
        plt.gca().set_aspect("equal")
        plt.title("Vogel spiral with %i spheres" % number_of_spheres)
        smuthi.postprocessing.graphical_output.plot_particles(
            -1e5, 1e5, -1e5, 1e5, 0, 0, spheres_list, 1000, False)
        plt.savefig("vogel_spiral_%i.png" % number_of_spheres)

    return [(ecs["top"] + ecs["bottom"]).real, preparation_time, solution_time]

# -----------------------------------------------------------------------------
# launch a series of simulations:

# iterative solution on GPU
gpu_iterative_particle_numbers = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
gpu_iterative_times = []
gpu_iterative_preptimes = []
gpu_iterative_ecs = []
for particle_number in gpu_iterative_particle_numbers:
    print("\n----------------------------------------------------------")
    print("Simulating %i particles on GPU with iterative solver."%particle_number)
    results = simulate_N_spheres(number_of_spheres=particle_number,
                                 direct_inversion=False,
                                 use_gpu=True)
    gpu_iterative_times.append(results[1]+results[2])
    gpu_iterative_preptimes.append(results[1])
    gpu_iterative_ecs.append(results[0])

# direct solution on CPU
cpu_direct_particle_numbers = [5, 10, 20, 50, 100]
cpu_direct_times = []
cpu_direct_preptimes = []
cpu_direct_ecs = []
for particle_number in cpu_direct_particle_numbers:
    print("\n----------------------------------------------------------")
    print("Simulating %i particles on CPU with direct inversion."%particle_number)
    results = simulate_N_spheres(number_of_spheres=particle_number,
                                 direct_inversion=True,
                                 use_gpu=False)
    cpu_direct_times.append(results[1]+results[2])
    cpu_direct_preptimes.append(results[1])
    cpu_direct_ecs.append(results[0])

# iterative solution on CPU
cpu_iterative_particle_numbers = [5, 10, 20, 50, 100, 200, 500]
cpu_iterative_times = []
cpu_iterative_preptimes = []
cpu_iterative_ecs = []
for particle_number in cpu_iterative_particle_numbers:
    print("\n----------------------------------------------------------")
    print("Simulating %i particles on CPU with iterative solver."%particle_number)
    results = simulate_N_spheres(number_of_spheres=particle_number,
                                 direct_inversion=False,
                                 use_gpu=False)
    cpu_iterative_times.append(results[1]+results[2])
    cpu_iterative_preptimes.append(results[1])
    cpu_iterative_ecs.append(results[0])

# -----------------------------------------------------------------------------
# plot the results

# get GPU device name
device_name = "no gpu"
try:
    import pycuda.driver as drv
    drv.init()
    device_name = drv.Device(0).name()
except:
    pass

# plot runtime
plt.figure()
plt.xlabel("Number of spheres")
plt.ylabel("Solver time")
plt.loglog(cpu_direct_particle_numbers, cpu_direct_times, '-bx')
plt.loglog(cpu_direct_particle_numbers, cpu_direct_preptimes, '--bx')
plt.loglog(cpu_iterative_particle_numbers, cpu_iterative_times, '-ro')
plt.loglog(cpu_iterative_particle_numbers, cpu_iterative_preptimes, '--ro')
plt.loglog(gpu_iterative_particle_numbers, gpu_iterative_times, '-gd')
plt.loglog(gpu_iterative_particle_numbers, gpu_iterative_preptimes, '--gd')
plt.legend(["direct, CPU, total", "direct, CPU, prep.", "iter., CPU, total", "iter., CPU, prep.", "iter., GPU, total",
            "iter., GPU, prep."])
plt.grid()
plt.savefig("runtime.png")

# plot extinction cross section
plt.figure()
plt.xlabel("Number of spheres")
plt.ylabel("Extinction cross section [nm^2]")
plt.loglog(cpu_direct_particle_numbers, cpu_direct_ecs, '-bx')
plt.loglog(cpu_iterative_particle_numbers, cpu_iterative_ecs, '-ro')
plt.loglog(gpu_iterative_particle_numbers, gpu_iterative_ecs, '-gd')
plt.legend(["Direct solution on CPU", "Iterative solution on CPU", "Iterative solution on " + device_name])
plt.grid()
plt.savefig("cross_section.png")

# create illustrations
# check if gpu can be enabled
cu.enable_gpu()
gpu_enabled = cu.use_gpu
results = simulate_N_spheres(number_of_spheres=200,
                             direct_inversion=False,
                             use_gpu=gpu_enabled,
                             make_illustrations=True)
