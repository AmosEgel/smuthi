# -*- coding: utf-8 -*-
"""
This script compares the accuracy of the scattering matrix scheme with the transfer matrix scheme and
highlights the role of extended precision arithmetics for problems with tunneling through thick layers.
"""

import numpy as np
import smuthi.layers as lay
import matplotlib.pyplot as plt

# Parameter input ----------------------------
vacuum_wavelength = 550
thicknesses = [0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 0]
n1 = 1
n2 = 2 + 0.01j
refractive_indices = [n1, n2, n1, n2, n1, n2, n1, n2, n1, n2, n1]
neff_refl = np.linspace(0, 10, 500)
neff_L = np.linspace(0, 2.6, 500)
omega = 2 * np.pi / vacuum_wavelength
kpar_refl = omega * neff_refl
kpar_L = omega * neff_L
pol = 0
# --------------------------------------------

# reflectivity of complete layer stack: compare scattering matrix vs transfer matrix approach
r_s = []
r_t = []
for kp in kpar_refl:
    s = lay.layersystem_scattering_matrix(pol, thicknesses, refractive_indices, kp, omega)
    t = lay.layersystem_transfer_matrix(pol, thicknesses, refractive_indices, kp, omega)

    r_s.append(s[1, 0])
    r_t.append(t[1, 0] / t[0, 0])

# excitation from inside: compare extended precision to standard precision
fromlayer = 4
tolayer = 6

L = lay.layersystem_response_matrix(pol, thicknesses, refractive_indices, kpar_L, omega, fromlayer, tolayer)
L50 = lay.layersystem_response_matrix(pol, thicknesses, refractive_indices, kpar_L, omega, fromlayer, tolayer, 50)

plt.figure()
plt.semilogy(neff_refl, abs(np.array(r_s)), color='red', label='scattering matrix')
plt.semilogy(neff_refl, abs(np.array(r_t)), 'b--', linewidth=2, label='transfer matrix')
plt.legend()
plt.xlabel('n_effective')
plt.ylabel('reflection coefficient')

plt.figure()
plt.semilogy(neff_L, abs(L[1, 1, :]), color='red', label='standard precision')
plt.semilogy(neff_L, abs(L50[1, 1, :]), 'b--', linewidth=2, label='50 digits precision')
plt.legend()
plt.xlabel('n_effective')
plt.ylabel('L_{1, 1}')
plt.show()

export_array = np.concatenate([abs(np.array(neff_refl))[:, None], abs(np.array(r_s))[:, None], abs(np.array(r_t))[:, None]], axis=1)
np.savetxt('scattering_vs_transfer_matrix.dat', export_array,
           header='columns are: n_eff, |r| for scattering matrx, |r| for transfer matrix')

export_array = np.concatenate([abs(np.array(neff_L))[:, None], abs(L[1, 1, :])[:, None], abs(L50[1, 1, :])[:, None]], axis=1)
np.savetxt('standard_vs_extended_precision.dat', export_array,
           header='columns are: n_eff, |L_11| for standard precision, |L_11| for 50 digits precision')