# This is an exemplary script to run SMUTHI from within python.
#
# It evaluates the scattering response of a Si or Au NP on Au
# substrate. The system is excited by a dipole source. This is a good
# approximation to the exitation of a plasmon with a scanning
# tunneling microscope (STM).

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
from smuthi.optical_constants import read_refractive_index_from_yaml as GetIndex
import sys
import os

smuthi.cuda_sources.enable_gpu()  # Enable GPU acceleration (if available)

#index_glass = 1.55 +1e-15j
index_glass = 1.55# +1e-15j
#spacer = 10 #nm
spacer = 5 #nm
samples = [
    ["c-5-STM00-v0", 2.73, 26.36],
    # ["e-4-STM02-20", 4.63, 16.27],
    # ["c-3-STM02-30", 4.01, 26.73],
    # ["b-2-STM02-50", 6.13, 43.1],
    # ["a-1-STM02-40", 5.56, 47.16]
]
def GetTopTSCS(WL, index_chrome, index_gold,l_max, neff_max, sample):
    #smuthi.layers.set_precision(1000)
    # Initialize the layer system object
    # The coordinate system is such that the interface between the
    # first two layers defines the plane z=0.
    thickness_chrome = sample[1]
    thickness_gold = sample[2]
    two_layers = smuthi.layers.LayerSystem(
        thicknesses=[0, thickness_gold, thickness_chrome,  0],         #  ambient, substrate
        refractive_indices=[1.0, index_gold, index_chrome, index_glass ])   # like glass, air



    # Define the scattering particles
    particle_grid = []
    sphere = smuthi.particles.Sphere(position=[0, 0, -spacer/2.0],
                                     refractive_index=index_glass,
                                     radius=spacer/2.0,
                                     l_max=l_max)    # choose l_max with regard to particle size and material
                                                 # higher means more accurate but slower
    particle_grid.append(sphere)

    sphere = smuthi.particles.Sphere(position=[0, 0, -spacer -core_r*2 -spacer*2 - stm_r],
                                     refractive_index=index_gold,
                                     radius=stm_r,
                                     l_max=l_max)    # choose l_max with regard to particle size and material
                                                 # higher means more accurate but slower
    particle_grid.append(sphere)


    # Define contour for Sommerfeld integral
    alpha = np.linspace(0, 2*np.pi, 100)

    smuthi.coordinates.set_default_k_parallel(vacuum_wavelength=WL,
            neff_resolution=1e-2
            , neff_max = neff_max
            # ,neff_waypoints=[0,0.9-0.15j, 8.7-0.15j, 8.7, neff_max]
                                                  )
    # smuthi.coordinates.set_default_k_parallel(vacuum_wavelength=WL,
    #        neff_resolution=1e-2,
    #        neff_waypoints=[0,0.9-0.15j, 1.1-0.15j, 1.2, neff_max] )
    dipole_source = smuthi.initial_field.DipoleSource(vacuum_wavelength=WL,
                           dipole_moment=dipole_moment,
                           position=[0, 0, -spacer -core_r*2 -spacer],
                           # position=[0, 0, -spacer],
                           azimuthal_angles=alpha)


    # Initialize and run simulation
    simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                              particle_list=particle_grid,
                                              initial_field=dipole_source,
                                              solver_type='LU',
                                              # solver_type='gmres',
                                              solver_tolerance=1e-5,
                                              store_coupling_matrix=True,
                                              coupling_matrix_lookup_resolution=None,
                                              # store_coupling_matrix=False,
                                              # coupling_matrix_lookup_resolution=5,
                                                  coupling_matrix_interpolator_kind='cubic'
                                                  ,log_to_file=False
                                                  ,log_to_terminal=False
                                                  )

    simulation.run()
    #polar = np.linspace(0, 0.228857671*np.pi, 100) # 0.21*pi rad == 37.8 grad
    polar = np.linspace(0, 0.21*np.pi, 100) # 0.21*pi rad == 37.8 grad

    total_far_field, initial_far_field, scattered_far_field = smuthi.scattered_field.total_far_field(
        initial_field=dipole_source, particle_list=particle_grid, layer_system=two_layers
        ,        polar_angles=polar
    )

    diss_pow = dipole_source.dissipated_power(particle_grid, two_layers)
    assert abs(diss_pow.imag / diss_pow) < 1e-8
    diss_pow = diss_pow.real
    top_pow, bottom_pow = 0, 0
    P0 = dipole_source.dissipated_power_homogeneous_background(layer_system=two_layers)
    top_pow = sum(total_far_field.integral()).real #sum both polarizations


    return   WL,  top_pow,  diss_pow, P0 , diss_pow/P0


output_directory = 'smuthi_output/dipole_source'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def main(l_max, neff_max,plt, sample):

    WLs = np.linspace(from_WL, to_WL, total_points)

    #index_Si = GetIndex('data/a-Si-Pierce-Palik.yml', WLs, "nm")
    index_Si = GetIndex('data/Si-Green-2008.yml', WLs, "nm")
    #index_Au = GetIndex('data/Au-Rakic-LD.yml', WLs, "nm")
    #index_Au = GetIndex('data/Au-Johnson.yml', WLs, "nm")
    #index_Au = GetIndex('data/Au-McPeak.yml', WLs, "nm")
    index_Au = GetIndex('data/Au-Olmon-ev.yml', WLs, "nm")
    index_Cr = GetIndex('data/Cr-Johnson.yml', WLs, "nm")


    val = []
    for i in range(len(WLs)):
    # for i in range(1):
    #     i=-1

        # use_gold=True
        # index_NP = index_Au[i][1]

        use_gold=False
        index_NP = index_Si[i][1]

        index_gold = index_Au[i][1]
        index_chrome = index_Cr[i][1]
        print("===> Params: l_max =", l_max, "neff_max =", neff_max, "  WL", WLs[i]," (",from_WL,"-",to_WL,
              ") ",core_r,"-> Au", index_gold,"-> Cr", index_chrome)
        valAu= GetTopTSCS(WLs[i], index_chrome, index_gold,l_max, neff_max, sample)
        sys.stdout = sys.__stdout__ # Restore output after muting it in simulation
        print(valAu)
        # index_NP = index_Si[i][1]
        # valSi = GetTopTSCS(WLs[i], index_NP, index_gold)
        # val.append(np.hstack((valAu, valSi)))
        val.append(np.hstack((valAu)))
    val = np.array(val)
    c = 3*10**5 # speed of light, nm*THz
    #return     1top_pow,     2bottom_pow ,    3absorbption , 4diss_pow, P0 , diss_pow/P0

    plt.plot( val[:,0], val[:,1]/val[:,3])
    #plt.ylim(0, 1.75e-8)
    plt.legend([r"transmitted/$P_{vac}$"])
    #plt.xlabel(r'THz')
    plt.xlabel(r'$\lambda, nm$')
    plt.ylabel(r'Share')
    gold = "Si"
    if use_gold:
        gold = "Au"
    sign_t = sample[0]+"_Cr"+str(sample[1])+"nm_Au"+str(sample[2])+"nm_spacer"+str(spacer)+"_"+str(from_WL)+"-dipole_lmax"+str(int(l_max))+"_sign"+sign+"_points"+str(total_points)+"_neff"+ str(neff_max)
    plt.title(#gold+' r = '+str(core_r)+"STM Au r="+str(stm_r)+"\n"+
              sign_t+"\n", fontsize = 10)
    plt.savefig(output_directory+"/dipole_tr-normP0_"+sign_t+".png")
    plt.clf()
    np.savetxt(output_directory+"/data_"+sign_t+".txt",val)


    plt.close('all')

#neff_max=10
#neff_list = [10,25,50]
neff_list = [5]
#neff_list = [5,10,20]
#neff_list = [25]
#neff_list = [10,50,70,100,200]
#neff_list = [50,70]
#neff_list = [100]
#l_max_list = [11,15,17,21,25]
#l_max_list = [3]
l_max_list = [7,9]
#l_max_list = [17]
dipole_moment=[0, 0, 1]
sign = str(dipole_moment[0])+str(dipole_moment[1])+str(dipole_moment[2])

core_r = 100.0
stm_r = 20.0
#total_points = 11
total_points = 81
total_points = 2

#from_WL = 400 # nm
from_WL = 800 # nm
to_WL = 1000 # nm,

#from_WL = 450 # nm
#to_WL = 650 # nm,

#plt.figure(figsize=(6,4))
#from_WL = 570
#to_WL = 1665 # nm, 180 THz

for neff_max in neff_list:
    for sample in samples:
        for l_max in l_max_list:
            main(l_max, neff_max,plt, sample)
            print("----")

            # l_max = 5
# main(l_max, neff_max,plt)
#plt.close()
# final = np.vstack((WLs,Q_sca)).T
# np.savetxt(output_directory+"/Q_sca_spectra.csv",final,header="WL,nm                  Q_sca  ")
plt.clf()
plt.close('all')

for neff_max in neff_list: 
    plt.figure(figsize=(6,4))
    for sample in samples:
        for l_max in l_max_list:
            sign_t = sample[0]+"_Cr"+str(sample[1])+"nm_Au"+str(sample[2])+"nm_spacer"+str(spacer)+"_"+str(from_WL)+"-dipole_lmax"+str(int(l_max))+"_sign"+sign+"_points"+str(total_points)+"_neff"+ str(neff_max)
            val = np.loadtxt(output_directory+"/data_"+sign_t+".txt")
            #main(l_max, neff_max,plt, sample)
            plt.plot( val[:,0], val[:,1]/val[:,3])
        #plt.ylim(0, 1.75e-8)
#plt.legend([r"transmitted/$P_{vac}$"])
#plt.xlabel(r'THz')
    plt.xlabel(r'$\lambda, nm$')
    plt.ylabel(r'Share')

    print("----")
    sign_t = "spacer"+str(spacer)+"_"+str(from_WL)+"-dipole_lmax"+str(int(l_max))+"_sign"+sign+"_points"+str(total_points)+"_neff"+ str(neff_max)
    plt.title(#gold+' r = '+str(core_r)+"STM Au r="+str(stm_r)+"\n"+
            sign_t+"\n", fontsize = 10)
    plt.savefig(output_directory+"/dipole_tr-normP0_"+sign_t+".png")
    plt.clf()
    plt.close
