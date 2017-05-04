# This is an exemplary script to run SMUTHI from within python.

import smuthi.simulation
import smuthi.coordinates
import smuthi.post_processing
import smuthi.index_conversion
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Set truncation parameters for spherical wave expansion:
smuthi.index_conversion.set_swe_specs(l_max=3)

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

simulation.initial_field_collection.vacuum_wavelength = vacuum_wavelength
simulation.initial_field_collection.add_planewave(amplitude=plane_wave_amplitude,
                                                  polar_angle=plane_wave_polar_angle,
                                                  azimuthal_angle=plane_wave_azimuthal_angle,
                                                  polarization=plane_wave_polarization)

# ----------------------------------------------------------------------------------------------------------------------
# Define the layer system
layer_thicknesses = [0, 500, 0]
layer_complex_refractive_indices = [1.5, 1.8 + 0.01j, 1]

simulation.layer_system.thicknesses = layer_thicknesses
simulation.layer_system.refractive_indices = layer_complex_refractive_indices

# ----------------------------------------------------------------------------------------------------------------------
# Define the scattering particles
sphere1_position = [100, 200, 300]
sphere1_radius = 120
sphere1_complex_refractive_index = 2.4 + 0.05j

sphere2_position = [-100, -300, 200]
sphere2_radius = 140
sphere2_complex_refractive_index = 2.2 + 0.01j

simulation.particle_collection.add_sphere(radius=sphere1_radius,
                                          refractive_index=sphere1_complex_refractive_index,
                                          position=sphere1_position)
simulation.particle_collection.add_sphere(radius=sphere2_radius,
                                          refractive_index=sphere2_complex_refractive_index,
                                          position=sphere2_position)

# ----------------------------------------------------------------------------------------------------------------------
# Define contour for Sommerfeld integral
contour_waypoints = [0, 0.8, 0.8 - 0.05j, 2.2 - 0.05j, 2.2, 5]
contour_discretization = 1e-3
complex_contour = smuthi.coordinates.ComplexContour(neff_waypoints=contour_waypoints,
                                                    neff_discretization=contour_discretization)

# ----------------------------------------------------------------------------------------------------------------------
# Run simulation and show some output
simulation.run()

scattering_cs = smuthi.post_processing.scattering_cross_section(
                                 initial_field_collection=simulation.initial_field_collection,
                                 particle_collection=simulation.particle_collection,
                                 linear_system=simulation.linear_system,
                                 layer_system=simulation.layer_system)

print('Total scattering cross section:')
print(scattering_cs['total top'][0] + scattering_cs['total top'][1])




