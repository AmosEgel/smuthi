#*****************************************************************************#
# This example script is an advanced version of the "dielectric sphere on a   #
# substrate" script. It demonstrates how Smuthi can be used to moldel TIR     #
# scattering microscopy (TIRSM).                                              #
#                                                                             #
# The purpose of this example is to study how the scattering signal from a    #
# plasmonic sphere on a glass substrate, excited by a totally reflected wave  #
# incident from the substrate side and collected by an objective with a given #
# numerical aperture depends on the wavelength of the incident wave and on    #
# the distance of the sphere to the substate surface.                         #
#*****************************************************************************#

import numpy as np
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.scattered_field
import smuthi.graphical_output


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
                                            polar_angle=-np.pi,  # from top
                                            azimuthal_angle=0,
                                            polarization=0)  # 0=TE 1=TM

# Initialize and run simulation
simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                          particle_list=one_sphere,
                                          initial_field=plane_wave)
simulation.run()

# Show differential scattering cross section
dscs = smuthi.scattered_field.scattering_cross_section(initial_field=plane_wave,
                                                       particle_list=one_sphere,
                                                       layer_system=two_layers)

smuthi.graphical_output.show_far_field(dscs, 
                                       save_plots=True, 
                                       show_plots=False, 
                                       outputdir='sphere_on_substrate')
