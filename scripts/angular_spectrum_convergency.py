'''In this script, the angular spectrum of a dipole source is computed, and compared to the spectrum generated from
first translating the source using the SVWF addition theorem and then computing the angular spectrum.'''

import numpy as np
import matplotlib.pyplot as plt
import smuthi.vector_wave_functions as vwf
import smuthi.coordinates as coord
import smuthi.spherical_functions as sf
import scipy.optimize
import json

k0 = 1
kp_array = np.linspace(0, 20, 2000)
kz_array = coord.k_z(k_parallel=kp_array, k=k0)
alpha = 0
pol = 0
dxmax = 6
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
    
    for idx, dx in enumerate(np.linspace(0.5, dxmax, 2 * dxmax)):
        pvwf_translation = np.exp(- 1j * kp_array * (np.cos(alpha) * dx + np.sin(alpha) * dy))
        dxlist.append(dx)
        kmaxlist.append([])
        lmaxlist.append([])
        dd = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        sph_bessel = [sf.spherical_bessel(n, k0 * dd) for n in range(llimit + 2)]
        
        costthetd = dz / dd
        sinthetd = np.sqrt(dx ** 2 + dy ** 2) / dd
        legendre, _, _ = sf.legendre_normalized(costthetd, sinthetd, llimit + 1)
        
        for lmax in range(12, llimit + 1, 5):
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
    with open('kmax_data.json', 'r') as fp:
        data = json.load(fp)
    dxlist = data['dxlist']
    kmaxlist = data['kmaxlist']
    lmaxlist = data['lmaxlist']
    g2list = data['g2list']
# save 
data = {'dxlist': dxlist, 'kmaxlist': kmaxlist, 'lmaxlist': lmaxlist, 'g2list': g2list}
with open('kmax_data.json', 'w') as fp:
    json.dump(data, fp)

# ----------------------------------------------------------------------------------------------------------------------
# linear fit
# kmax = a * lmax + b

# compute a and b for each distance
alist = []
blist = []

for i, dx in enumerate(dxlist):
    blist.append((kmaxlist[i][-1] - kmaxlist[i][0]) / (lmaxlist[i][-1] - lmaxlist[i][0]))
    alist.append(kmaxlist[i][-1] - blist[-1] * lmaxlist[i][-1])

plt.figure()
plt.semilogy(dxlist, blist)

plt.figure()
plt.plot(dxlist, alist)


# Assume that a is constant with the translation distance, whereas b depends 
# on the distance like this:

def b_model(dist, b0, beta):
    return b0 * np.exp(- beta * dist)

# Fit this model to the measured b data
b_params, _ = scipy.optimize.curve_fit(b_model, np.array(dxlist), np.array(blist), p0=(1,1))

print(b_params)

b0fit = b_params[0]
betafit = b_params[1]



"""
def kmax_model(beta, a, dx, lmaxarray):
    return np.exp(- beta * dx) * np.array(lmaxarray) + a - np.array(kmaxlist[i])    

def residual(params)    :
    beta = params[0]
    a = params[1]
    resid = 0
    for i, dx in enumerate(dxlist):   
        resid += abs(kmax_model(beta, a, dx, np.array(lmaxlist[i])) - np.array(kmaxlist[i])) ** 2
    return resid


fit_params = scipy.optimize.leastsq(residual, [0, 0])
beta_fit = fit_params[0][0]
a_fit = fit_params[0][1]
print(beta_fit, a_fit)
"""    
    
    
afit = sum(alist) / len(alist)
print(afit)



# Plot results

#plt.figure()
#plt.plot(kp_array, abs(g))
#plt.plot(kp_array, abs(g2), 'r--')
#plt.ylim((0, 3))

#plt.figure()
#for ig2, g2 in enumerate(g2list):
#    plt.semilogy(kp_array, abs(g - g2) / abs(g), label=ig2)
#plt.legend()

plt.figure()
for i, dx in enumerate(dxlist):
    plt.plot(lmaxlist[i], kmaxlist[i], label=dx)
plt.legend()    

for i, dx in enumerate(dxlist):
    #bfit = b_model(dx, *b_params)
    bfit = b0fit * np.exp(- betafit * dx)
    plt.plot(lmaxlist[i], afit + bfit * np.array(lmaxlist[i]), 'o')

plt.show()





