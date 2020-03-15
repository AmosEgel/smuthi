#*****************************************************************************#
# This is a simple example script for Smuthi v10.0.0                          #
# It evaluates the total scattering cross section of a glass sphere on a      #
# glass substrate, excited by a plane wave under normal incidence.            #
#*****************************************************************************#

import numpy as np
import matplotlib.pyplot as plt
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.far_field
import smuthi.postprocessing.graphical_output


# In this file, all lengths are given in nanometers

# Initialize the layer system object containing the substrate (glass) half 
# space and the ambient (air) half space. The coordinate system is such that 
# the interface between the first two layers defines the plane z=0.
# Note that semi infinite layers have thickness 0!
two_layers = smuthi.layers.LayerSystem(thicknesses=[0, 0],
                                       refractive_indices=[1.52, 1])

# Scattering particle
sphere = smuthi.particles.Sphere(position=[0, 0, 100],
                                 refractive_index=1.52,
                                 radius=100,
                                 l_max=3)

# list of all scattering particles (only one in this case)
one_sphere = [sphere]

# Initial field
plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=550,
                                            polar_angle=np.pi,    # from top
                                            azimuthal_angle=0,
                                            polarization=0)       # 0=TE 1=TM

# Initialize and run simulation
simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                          particle_list=one_sphere,
                                          initial_field=plane_wave)
simulation.run()

# evaluate the scattering cross section
scs = smuthi.postprocessing.far_field.total_scattering_cross_section(initial_field=plane_wave,
                                                                     particle_list=one_sphere,
                                                                     layer_system=two_layers)

print("\n****************************************************")
print("Scattering cross section: %e µm^2"%(scs/1e6))
print("****************************************************")