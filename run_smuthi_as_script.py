# This is an exemplary script to run SMUTHI from within python.

import numpy as np
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.coordinates
import smuthi.post_processing
import smuthi.field_evaluation

# ----------------------------------------------------------------------------------------------------------------------
# Initialize simulation object
simulation = smuthi.simulation.Simulation()

# ----------------------------------------------------------------------------------------------------------------------
# Define the vacuum wavelength and the initial field
vacuum_wavelength = 550
plane_wave_polar_angle = 30 * np.pi / 180  # polar angle of the incoming plane wave in radian
plane_wave_azimuthal_angle = 45 * np.pi / 180  # azimuthal angle of the incoming plane wave in radian
plane_wave_amplitude = 1
plane_wave_polarization = 0  # 0 stands for TE, 1 stands for TM

plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=vacuum_wavelength, polar_angle=plane_wave_polar_angle, 
                                               azimuthal_angle=plane_wave_azimuthal_angle, 
                                               polarization=plane_wave_polarization)
simulation.initial_field = plane_wave

# ----------------------------------------------------------------------------------------------------------------------
# Define the layer system
layer_thicknesses = [0, 500, 0]
layer_complex_refractive_indices = [1.5, 1.8 + 0.01j, 1]

layer_system = smuthi.layers.LayerSystem(thicknesses=layer_thicknesses, 
                                         refractive_indices=layer_complex_refractive_indices)
simulation.layer_system = layer_system
    
# ----------------------------------------------------------------------------------------------------------------------
# Define the scattering particles

sphere1 = smuthi.particles.Sphere(position=[100, 200, 300], refractive_index=2.4+0.05j, radius=120, l_max=3)
sphere2 = smuthi.particles.Sphere(position=[-100, -300, 200], refractive_index=2.2+0.01j, radius=140, l_max=3)
simulation.particle_list = [sphere1, sphere2]

# ----------------------------------------------------------------------------------------------------------------------
# Define contour for Sommerfeld integral
contour_waypoints = [0, 0.8, 0.8 - 0.05j, 2.2 - 0.05j, 2.2, 5]
contour_discretization = 1e-3
simulation.wr_neff_contour = smuthi.coordinates.ComplexContour(neff_waypoints=contour_waypoints,
                                                               neff_discretization=contour_discretization)

# ----------------------------------------------------------------------------------------------------------------------
# Run simulation and show some output
simulation.run()

scs = smuthi.field_evaluation.scattering_cross_section(initial_field=simulation.initial_field,
                                                       particle_list=simulation.particle_list,
                                                       layer_system=simulation.layer_system)

print('Total scattering cross section:')
print(sum(scs[0].integral()))




