import numpy as np
import smuthi.initial_field as init
import smuthi.layers
import smuthi.particles
import smuthi.fields

ld = 532
A = 1
beta = 0
alpha = 0.2 * np.pi
pol = 1
rS = [100, -200, 300]
laysys = smuthi.layers.LayerSystem(thicknesses=[0, 0], refractive_indices=[1, 1])
particle = smuthi.particles.Sphere(position=rS, l_max=3, m_max=3)
particle_list = [particle]
al_ar = np.linspace(0, 2*np.pi, 1000)
kp_ar = np.linspace(0, 0.99999, 1000) * smuthi.fields.angular_frequency(ld)
bw = 4000
ref = [-100, 100, 200]
gauss_beam = init.GaussianBeam(vacuum_wavelength=ld, polar_angle=beta, azimuthal_angle=alpha, polarization=pol,
                               amplitude=A, reference_point=ref, k_parallel_array=kp_ar, 
                               beam_waist=bw)
particle.initial_field = gauss_beam.spherical_wave_expansion(particle, laysys)


def test_SWE_coefficients_against_prototype():
    aI = particle.initial_field.coefficients
    ai0 = 0.4040275 - 1.6689055j
    ai9 = -0.0115840 + 0.0107125j
    ai20 = -3.1855342e-04 - 7.7089434e-04j
    print(abs((aI[0] - ai0) / ai0), abs((aI[9] - ai9) / ai9), abs((aI[20] - ai20) / ai20))
    assert abs((aI[0] - ai0) / ai0) < 1e-5
    assert abs((aI[9] - ai9) / ai9) < 1e-5
    assert abs((aI[20] - ai20) / ai20) < 1e-5
    
    
def test_focus_field():
    E = gauss_beam.electric_field(np.array([ref[0]]), np.array([ref[1]]), np.array([ref[2]]), laysys)
    print(abs(np.sqrt(E[0]**2 + E[1]**2 + E[2]**2) - A))
    assert abs(np.sqrt(E[0]**2 + E[1]**2 + E[2]**2) - A) < 1e-3
    

if __name__ == '__main__':
    test_SWE_coefficients_against_prototype()
    test_focus_field()
