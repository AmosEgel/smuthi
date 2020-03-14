# This is an exemplary script to run SMUTHI from within python.
# It evaluates the scattering response of a finite periodic grid of dielectric spheres that are located on a metallic
# substrate coated with a dielectric layer. The system is excited by a plane wave under normal incidence.

import numpy as np
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.graphical_output
import smuthi.postprocessing.scattered_field
import smuthi.utility.cuda

smuthi.utility.cuda.enable_gpu()  # Enable GPU acceleration (if available)

# Initialize a plane wave object the initial field
plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=550,
                                            polar_angle=-np.pi,       # normal incidence, from top
                                            azimuthal_angle=0,
                                            polarization=0)           # 0 stands for TE, 1 stands for TM

# Initialize the layer system object
# The coordinate system is such that the interface between the first two layers defines the plane z=0.
three_layers = smuthi.layers.LayerSystem(thicknesses=[0, 50, 0],               # substrate, dielectric layer, ambient
                                         refractive_indices=[1+6j, 1.49, 1])   # like aluminum, SiO2, air

# Define the scattering particles
particle_grid = []
for x in range(-1500, 1501, 750):
    for y in range(-1500, 1501, 750):
        sphere = smuthi.particles.Sphere(position=[x, y, 150],
                                         refractive_index=2.4,
                                         radius=100,
                                         l_max=3)    # choose l_max with regard to particle size and material
                                                     # higher means more accurate but slower
        particle_grid.append(sphere)

# Initialize and run simulation
simulation = smuthi.simulation.Simulation(layer_system=three_layers,
                                          particle_list=particle_grid,
                                          initial_field=plane_wave,
                                          solver_type='gmres',
                                          solver_tolerance=1e-3,
                                          store_coupling_matrix=False,
                                          coupling_matrix_lookup_resolution=5,
                                          coupling_matrix_interpolator_kind='cubic')
simulation.run()

# Show the far field
scattered_far_field = smuthi.postprocessing.scattered_field.scattered_far_field(
    vacuum_wavelength=plane_wave.vacuum_wavelength,
    particle_list=simulation.particle_list,
    layer_system=simulation.layer_system)

output_directory = 'smuthi_output/smuthi_as_script'

smuthi.postprocessing.graphical_output.show_far_field(far_field=scattered_far_field,
                                                       save_plots=True,
                                                       show_plots=False,
                                                       outputdir=output_directory+'/far_field_plots')

# Show the near field
smuthi.postprocessing.graphical_output.show_near_field(quantities_to_plot=['E_y', 'norm_E', 'E_scat_y', 'norm_E_scat'],
                                                       save_plots=True,
                                                       show_plots=False,
                                                       save_animations=True,
                                                       outputdir=output_directory+'/near_field_plots',
                                                       xmin=-1700,
                                                       xmax=1700,
                                                       ymin=10,
                                                       ymax=10,
                                                       zmin=-100,
                                                       zmax=2200,
                                                       resolution_step=100,
                                                       interpolate_step=20,
                                                       simulation=simulation,
                                                       max_field=1.5)
