#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:56:50 2020

@author: eli
"""

import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import csv
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.graphical_output
import smuthi.postprocessing.scattered_field

pre_wavelengths = []
pre_ns = []
pre_ks = []
with open("Refractive_Info/Si_Aspnes.csv", newline='') as file:
    reader = csv.reader(file, delimiter=',', quotechar='|')
    count = 0;
    for row in reader:
        if count > 0:
            [w, n, k] = map(float, row)
            pre_wavelengths.append(1000*w)
            pre_ns.append(n)
            pre_ks.append(k)
        count += 1
n_predictor = interp.interp1d(pre_wavelengths, pre_ns)
k_predictor = interp.interp1d(pre_wavelengths, pre_ks)

test_l = np.linspace(min(pre_wavelengths), max(pre_wavelengths), 1000)
test_n = n_predictor(test_l)
test_k = k_predictor(test_l)
plt.plot(test_l, test_n, 'r')
plt.plot(test_l, test_k, 'b')
plt.show()
wavelengths = list(range(300,801,5))
rad = 44

# Get all particle examples
"""
particles = [smuthi.particles.Sphere(position=[0,0,0],
                                    refractive_index=n_predictor(wl) + 1j*k_predictor(wl),
                                    radius=rad,
                                    l_max=1
                                    ) for wl in wavelengths]
"""
particles = None

for i, wl in enumerate(wavelengths[1:15]+wavelengths[16:]):# + wavelengths[0:1]):
    #print(i, wl)
    #print("AAAA")
    # We want layer system to just be air extending infinitely
    print(i)
    #particles = None
    particles = smuthi.particles.Sphere(position=[0,0,0],
                                    refractive_index=n_predictor(wl) + 1j*k_predictor(wl),
                                    radius=rad,
                                    l_max=1
                                    )
    layers = None
    layers = smuthi.layers.LayerSystem(thicknesses=[0,0], refractive_indices=[1,1])
    plane_wave = None
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wl,
                                            polar_angle=-np.pi,       # normal incidence, from top
                                            azimuthal_angle=0,
                                            polarization=0)
    simulation = None
    simulation = smuthi.simulation.Simulation(layer_system=layers,
                                          particle_list=[particles],
                                          initial_field=plane_wave,
                                          )
    #simulation.set_logging(log_to_terminal = True)
#print("Oh")
    simulation.run()
    print(particles.scattered_field.coefficients_tlm(1,1,1))
    #plt.plot()
    #Some Postprocessing here
    #0.42237216222850177+0.23034090086333728j
    
