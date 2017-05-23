'''In this script, the angular spectrum of a dipole source is computed, and compared to the spectrum generated from
first translating the source using the SVWF addition theorem and then computing the angular spectrum.'''

import numpy as np
import matplotlib.pyplot as plt
import smuthi.vector_wave_functions as vwf
import smuthi.coordinates as coord
import smuthi.spherical_functions as sf
import scipy.optimize
import scipy.io

k0 = 1
kp_array = np.linspace(0, 20, 2000)
kz_array = coord.k_z(k_parallel=kp_array, k=k0)
alpha = 0
pol = 0
dxmax = 5
dy = 0
dz = 0
llimit = 20
compute = True

# ----------------------------------------------------------------------------------------------------------------------
# Angular spectrum of dipole source
l = 1
m = 0
tau = 0

B = vwf.transformation_coefficients_VWF(tau, l, m, pol, kp=kp_array, kz=-kz_array)
g = 1 / (2 * np.pi * kz_array * k0) * np.exp(1j * m * alpha) * B

# ----------------------------------------------------------------------------------------------------------------------
# Translated dipole source

g2list = []
lmaxlist = []
kmaxlist = []
dxlist = []

if compute:

    ct = kz_array / k0
    st = kp_array / k0
    plm_list, pilm_list, taulm_list = sf.legendre_normalized(ct, st, llimit)
    
    dstart = 1
    dstop = dxmax
    dstep = 0.25
    numd = (dxmax - dstart) / dstep + 1
    
    for idx, dx in enumerate(np.linspace(dstart, dstop, numd, endpoint=True)):
        pvwf_translation = np.exp(- 1j * kp_array * (np.cos(alpha) * dx + np.sin(alpha) * dy))
        dxlist.append(dx)
        kmaxlist.append([])
        lmaxlist.append([])
        dd = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        sph_bessel = [sf.spherical_bessel(n, k0 * dd) for n in range(llimit + 2)]
        
        costthetd = dz / dd
        sinthetd = np.sqrt(dx ** 2 + dy ** 2) / dd
        legendre, _, _ = sf.legendre_normalized(costthetd, sinthetd, llimit + 1)
        
        for lmax in range(1, llimit + 1):
            print(lmax)
            lmaxlist[-1].append(lmax)
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
        
            kmaxlist[-1].append(kp_array[kmax_idx])
else:
    # load data from .mat file
    data = scipy.io.loadmat('kmax_data.mat')
    dxlist = np.squeeze(data['dxlist'])
    kmaxlist = data['kmaxlist']
    lmaxlist = data['lmaxlist']
    g2list = data['g2list']

# save 
data = {'dxlist': dxlist, 'kmaxlist': kmaxlist, 'lmaxlist': lmaxlist, 'g2list': g2list}
scipy.io.savemat('kmax_data.mat', data)


plt.figure()
#for ig2, g2 in enumerate(g2list):
#    plt.semilogy(kp_array, abs(g - g2) / abs(g), label=ig2)
#plt.legend()
plt.semilogy(kp_array, abs(g - g2list[-1]) / abs(g))


# ----------------------------------------------------------------------------------------------------------------------
# linear fit
# kmax = a * lmax + b

# compute a and b for each distance
alist = []
blist = []

for i, dx in enumerate(dxlist):
    print(i, dx)
    blist.append((kmaxlist[i][-1] - kmaxlist[i][-2]) / (lmaxlist[i][-1] - lmaxlist[i][-2]))
    alist.append(kmaxlist[i][-1] - blist[-1] * lmaxlist[i][-1])
    

# Assume that a is constant with the translation distance, whereas b depends 
# on the distance like this:

#def b_model(dist, b0, beta):
#    return b0 * np.exp(- beta * dist)

def b_model(dist, b0):
    return b0 / dist

# Fit this model to the measured b data
b_params, _ = scipy.optimize.curve_fit(b_model, np.array(dxlist), np.array(blist), p0=(1))


b0fit = b_params[0]
afit = sum(alist) / len(alist)

plt.figure()
plt.plot(dxlist, np.array(blist), '.')
plt.plot(dxlist, b_model(dxlist, b0fit))

plt.ylabel('b coefficient')
plt.xlabel('translation distance')


plt.figure()
plt.plot(dxlist, alist, '.')
plt.plot(dxlist, [afit] * len(dxlist))
plt.ylabel('a coefficient')
plt.xlabel('translation distance')


plt.figure()
for i, dx in enumerate(dxlist):
    plt.plot(lmaxlist[i], kmaxlist[i], 'o', label=dx)
plt.legend()    
plt.ylabel('kmax')
plt.xlabel('lmax')

plt.gca().set_color_cycle(None)

for i, dx in enumerate(dxlist):
    #bfit = b_model(dx, *b_params)
    bfit = b_model(dx, b_params[0])
    plt.plot(lmaxlist[i], afit + bfit * np.array(lmaxlist[i]))

plt.show()

