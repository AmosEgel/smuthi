# -*- coding: utf-8 -*-

import layerresponse as lr
import matplotlib.pyplot as plt
import numpy as np

layer_d = [0,300,1000,500,0]
layer_n = [1,2+0j,3,2,1]

wl = 550
omega = 2*np.pi/wl
neff_array = np.linspace(0, 4, 508)
kpar_array = neff_array*omega
l11_array = []
pol=0
fromlayer = 3
tolayer = 1

#print(lr.layersystem_transfer_matrix(pol, layer_d, layer_n, 0, omega))

#print(lr.layersystem_scattering_matrix(pol, layer_d, layer_n, 0, omega))

#print(lr.layersystem_response_matrix(pol, layer_d, layer_n, 0, omega, fromlayer, tolayer))

for kpar in kpar_array:
    lmat = lr.layersystem_response_matrix(pol,layer_d,layer_n,kpar,omega,fromlayer, tolayer)
    l11_array.append(np.linalg.norm(lmat))

plt.semilogy(neff_array,l11_array)
plt.show()

