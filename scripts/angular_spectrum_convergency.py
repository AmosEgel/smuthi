'''In this script, the angular spectrum of a dipole source is computed, and compared to the spectrum generated from
first translating the source using the SVWF addition theorem and then computing the angular spectrum.'''

import numpy as np
import matplotlib.pyplot as plt
import smuthi.vector_wave_functions as vwf
import smuthi.coordinates as coord
import smuthi.spherical_functions as sf

k0 = 1
kp_array = np.linspace(0, 10, 100)
kz_array = coord.k_z(k_parallel=kp_array, k=k0)
alpha = 0
pol = 0
dx = 2
dy = 0
dz = 0
lmax = 15
# ----------------------------------------------------------------------------------------------------------------------
# Angular spectrum of dipole source
l = 1
m = 0
tau = 0

B = vwf.transformation_coefficients_VWF(tau, l, m, pol, kp=kp_array, kz=-kz_array)
g = 1 / (2 * np.pi * kz_array * k0) * np.exp(1j * m * alpha) * B

# ----------------------------------------------------------------------------------------------------------------------
# Translated dipole source

pvwf_translation = np.exp(- 1j * kp_array * (np.cos(alpha) * dx + np.sin(alpha) * dy))

g2list = []
lmaxlist = []
kmaxlist = []

ct = kz_array / k0
st = kp_array / k0
plm_list, pilm_list, taulm_list = sf.legendre_normalized(ct, st, lmax)

dd = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
sph_bessel = [sf.spherical_bessel(n, k0 * dd) for n in range(lmax + 2)]

costthetd = dz / dd
sinthetd = np.sqrt(dx ** 2 + dy ** 2) / dd
legendre, _, _ = sf.legendre_normalized(costthetd, sinthetd, lmax + 1)

for lmax in range(5, lmax + 1):
    print(lmax)
    lmaxlist.append(lmax)
    g2list.append(g - g)

    for tau2 in range(2):
        for l2 in range(1, lmax + 1):
            for m2 in range(-l2, l2 + 1):
                B2 = vwf.transformation_coefficients_VWF(tau2, l2, m2, pol, kp=kp_array, kz=-kz_array,
                                                         pilm_list=pilm_list, taulm_list=taulm_list)
                AB = vwf.translation_coefficients_svwf_out_to_out(l, m, l2, m2, k0, [dx, dy, dz], legendre=legendre,
                                                                  sph_bessel=sph_bessel)
                if tau == tau2:
                    g2list[-1] += 1 / (2 * np.pi * kz_array * k0) * np.exp(1j * m2 * alpha) * B2 * pvwf_translation * AB[0]
                else:
                    g2list[-1] += 1 / (2 * np.pi * kz_array * k0) * np.exp(1j * m2 * alpha) * B2 * pvwf_translation * AB[1]

    kmax_idx = 0
    for i in range(1, len(g2list[-1])):
        if abs(g2list[-1][i] - g[i]) / abs(g[i]) > 1:
            kmax_idx = i - 1
            print('break at ', i)
            break

    kmaxlist.append(kp_array[kmax_idx])


# ----------------------------------------------------------------------------------------------------------------------
# Plot results



plt.figure()
plt.plot(kp_array, abs(g))
#plt.plot(kp_array, abs(g2), 'r--')
plt.ylim((0, 3))

plt.figure()
for ig2, g2 in enumerate(g2list):
    plt.semilogy(kp_array, abs(g - g2) / abs(g), label=ig2)

plt.legend()

plt.figure()
plt.plot(lmaxlist, kmaxlist)

plt.show()