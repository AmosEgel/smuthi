# This is an exemplary script to run SMUTHI from within python.
#
# It evaluates the scattering response of a single silver NP
# in vacuum. The system is excited by a plane wave.

import matplotlib.pyplot as plt
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


def GetTSCS(WL, core_r, index_NP, samples):
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
                                     refractive_index=index_NP,
                                     radius=core_r,
                                     l_max=3)    # choose l_max with regard to particle size and material
                                                 # higher means more accurate but slower
    particle_grid.append(sphere)

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
                                              solver_tolerance=1e-5,
                                              store_coupling_matrix=True,
                                              coupling_matrix_lookup_resolution=None,
                                              # store_coupling_matrix=False,
                                              # coupling_matrix_lookup_resolution=5,
                                                  coupling_matrix_interpolator_kind='cubic')
    simulation.run()


    p_angles = np.linspace(0, np.pi, samples, dtype=float)
    a_angles = np.linspace(0, 2.0*np.pi, samples, dtype=float)
    #print(angles)
    scattering_cross_section = smuthi.scattered_field.scattering_cross_section(
        initial_field=plane_wave,
        particle_list=particle_grid,
        layer_system=two_layers
        ,polar_angles=p_angles
        ,azimuthal_angles=a_angles
        )
    Q_sca = (scattering_cross_section.top().integral()[0]
            + scattering_cross_section.top().integral()[1]
            + scattering_cross_section.bottom().integral()[0]
            + scattering_cross_section.bottom().integral()[1]).real/ (np.pi*core_r**2)
    return Q_sca


# WL=354 #nm
# core_r = WL/20.0
# index_NP = 4.0
# GetTSCS(WL,core_r,index_NP,samples)
#Q_sca exact value 0.01988453 (from Scattnlay)
integral_samples = 180 # Q_sca 0.0197530999065
#integral_samples = 1800 # Q_sca 0.0198715254489

core_r = 75
index_NP = 4.0

from_WL = 400
to_WL = 800
WLs = np.linspace(from_WL, to_WL, 101)
Q_sca = []
for WL in WLs:
    Q_sca.append(GetTSCS(WL,core_r,index_NP,integral_samples))
    print("\nWL =", WL,"(from", WLs[0],"to",WLs[-1],")  Q_sca = ", Q_sca[-1],"\n")
output_directory = 'smuthi_output/smuthi_as_script'
print(Q_sca)
plt.plot( WLs, Q_sca)
plt.xlabel(r'$\lambda, nm$')
plt.ylabel(r'$Q_{sca}$')
plt.title('Smuthi D = '+str(core_r*2)+"nm, index = "+str(index_NP))
plt.savefig(output_directory+"/Q_sca_spectra.png")

final = np.vstack((WLs,Q_sca)).T
np.savetxt(output_directory+"/Q_sca_spectra.csv",final,header="WL,nm                  Q_sca  ")

