# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import layerresponse_mp as lr
import cProfile
import timeit

layer_d = [0,3000,5000,5000,4000,0]
layer_n = [1,2+0j,3,2,2+0.001j,1]

wl = 550
omega = 2*np.pi/wl
neff_array = np.linspace(0, 10, 208)
kpar_array = neff_array*omega
l11_array = []
pol=0
fromlayer = 3
tolayer = 1

#print(lr.layersystem_transfer_matrix(pol, layer_d, layer_n, 0, omega))

#print(lr.layersystem_scattering_matrix(pol, layer_d, layer_n, 0, omega))


# def lrm():
#     lr.layersystem_response_matrix(pol, layer_d, layer_n, 0, omega, fromlayer, tolayer, 100)

# cProfile.run('lrm()')

# tm = timeit.timeit(lrm, number=200)
# print(tm)

for kpar in kpar_array:
    lmat = lr.layersystem_response_matrix(pol,layer_d,layer_n,kpar,omega,fromlayer, tolayer,500)
    l11_array.append(np.linalg.norm(lmat))

plt.semilogy(neff_array,l11_array)
plt.show()

