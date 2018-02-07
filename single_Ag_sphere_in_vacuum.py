# This is an exemplary script to run SMUTHI from within python.
#
# It evaluates the scattering response of a single silver NP
# in vacuum. The system is excited by a plane wave.

import numpy as np
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.coordinates
import smuthi.cuda_sources
import smuthi.scattered_field
import smuthi.graphical_output


smuthi.cuda_sources.enable_gpu()  # Enable GPU acceleration (if available)

WL=354 #nm
core_r = WL/20.0
epsilon_Ag = -2.0 + 0.28j
index_Ag = np.sqrt(epsilon_Ag)

# Initialize a plane wave object the initial field
plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=WL,
                                            polar_angle=-np.pi,       # normal incidence, from top
                                            azimuthal_angle=0,
                                            polarization=1)           # 0 stands for TE, 1 stands for TM

# Initialize the layer system object
# The coordinate system is such that the interface between the first two layers defines the plane z=0.
two_layers = smuthi.layers.LayerSystem(thicknesses=[0, 0],               # substrate,  ambient
                                         refractive_indices=[1.0, 1.0])   # like aluminum, SiO2, air

# Define the scattering particles
particle_grid = []
spacer = 1.1*core_r #nm
sphere = smuthi.particles.Sphere(position=[0, 0, 40], #core_r+spacer],
        refractive_index=index_Ag,
        radius=core_r,
                                 l_max=3)    # choose l_max with regard to particle size and material
                    # higher means more accurate but slower
particle_grid.append(sphere)
# sphere = smuthi.particles.Sphere(position=[0, 220, radius+spacer],
#         refractive_index=4.3,
#         radius=radius,
#         l_max=3)    # choose l_max with regard to particle size and material
#                     # higher means more accurate but slower
# particle_grid.append(sphere)

# Define contour for Sommerfeld integral
smuthi.coordinates.set_default_k_parallel(vacuum_wavelength=plane_wave.vacuum_wavelength,
        neff_resolution=5e-3,       # smaller value means more accurate but slower
        neff_max=2)                 # should be larger than the highest refractive
                                    # index of the layer system

# Initialize and run simulation
simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                          particle_list=particle_grid,
                                          initial_field=plane_wave,
                                          solver_type='LU',
                                          # solver_type='gmres',
                                          solver_tolerance=1e-3,
                                          store_coupling_matrix=True,
                                          coupling_matrix_lookup_resolution=None,
                                          # store_coupling_matrix=False,
                                          # coupling_matrix_lookup_resolution=5,
                                          coupling_matrix_interpolator_kind='cubic')
simulation.run()

# Show the far field
scattered_far_field = smuthi.scattered_field.scattered_far_field(
    vacuum_wavelength=plane_wave.vacuum_wavelength,
    particle_list=simulation.particle_list,
    layer_system=simulation.layer_system)

output_directory = 'smuthi_output/smuthi_as_script'

smuthi.graphical_output.show_far_field(far_field=scattered_far_field,
                                       save_plots=True,
                                       show_plots=False,
                                       outputdir=output_directory+'/far_field_plots')

# Show the near field
smuthi.graphical_output.show_near_field(quantities_to_plot=['E_y', 'norm_E', 'E_scat_y', 'norm_E_scat'],
                                        save_plots=True,
                                        show_plots=False,
                                        save_animations=False,
                                        outputdir=output_directory+'/near_field_plots_yz',
                                        xmin=0,
                                        xmax=0,
                                        ymin=-40,
                                        ymax=40,
                                        zmin=-0,
                                        zmax=80,
                                        resolution_step=2,
                                        interpolate_step=None,
                                        interpolation_order=1,
                                        simulation=simulation
                                        , max_field=8.3
                                        , min_norm_field=0.87
                                            )

smuthi.graphical_output.show_near_field(quantities_to_plot=['E_y', 'norm_E', 'E_scat_y', 'norm_E_scat'],
                                        save_plots=True,
                                        show_plots=False,
                                        save_animations=False,
                                        outputdir=output_directory+'/near_field_plots_xz',
                                        xmin=-40,
                                        xmax=40,
                                        ymin=0,
                                        ymax=0,
                                        zmin=-0,
                                        zmax=80,
                                        resolution_step=2,
                                        interpolate_step=None,
                                        interpolation_order=1,
                                        simulation=simulation
                                        , max_field=15.36
                                        , min_norm_field=0.82
                                            )
