# -*- coding: utf-8 -*-
import yaml
import smuthi.simulation
import numpy as np
import smuthi.index_conversion as idx
import smuthi.coordinates as coord


def read_input_yaml(filename):
    with open(filename, 'r') as input_file:
        input_data = yaml.load(input_file.read())

    simulation = smuthi.simulation.Simulation()

    simulation.initial_field_collection.vacuum_wavelength = float(input_data['vacuum wavelength'])

    for prtcl in input_data['scattering particles']:
        if prtcl['shape'] == 'sphere':
            r = float(prtcl['radius'])
            n = (float(prtcl['refractive index']) + 1j * float(prtcl['extinction coefficient']))
            pos = [float(prtcl['position'][0]), float(prtcl['position'][1]), float(prtcl['position'][2])]
            simulation.particle_collection.add_sphere(radius=r, refractive_index=n, position=pos)
        else:
            raise ValueError('Currently, only spheres are implemented')

    thick = [float(d) for d in input_data['layer system'][0]['thicknesses']]
    ref_ind = [float(n) for n in input_data['layer system'][0]['refractive indices']]
    ext_coeff = [float(n) for n in input_data['layer system'][0]['extinction coefficients']]
    ref_ind = np.array(ref_ind) + 1j * np.array(ext_coeff)
    ref_ind = ref_ind.tolist()
    simulation.layer_system.__init__(thicknesses=thick, refractive_indices=ref_ind)

    for infld in input_data['initial field']:
        if infld['type'] == 'plane wave':
            a = float(infld['amplitude'])
            if infld['angle units'] == 'degree':
                ang_fac = np.pi / 180
            else:
                ang_fac = 1
            pol_ang = ang_fac * float(infld['polar angle'])
            az_ang = ang_fac * float(infld['azimuthal angle'])
            if infld['polarization'] == 'TE':
                pol = 0
            elif infld['polarization'] == TM:
                pol = 1
            else:
                raise ValueError('polarization must be "TE" or "TM"')
            ref = [float(infld['reference point'][0]), float(infld['reference point'][1]),
                   float(infld['reference point'][2])]
            simulation.initial_field_collection.add_planewave(amplitude=a, polar_angle=pol_ang, reference_point=ref,
                                                              azimuthal_angle=az_ang, polarization=pol)

    lmax = int(input_data['lmax'])
    mmax = int(input_data['mmax'])
    simulation.linear_system.swe_specs = idx.swe_specifications(lmax, mmax)

    neff_waypoints = [complex(nf) for nf in input_data['neff waypoints']]
    neff_discretization = float(input_data['neff discretization'])
    simulation.wr_neff_contour = coord.ComplexContour(neff_waypoints=neff_waypoints,
                                                      neff_discretization=neff_discretization)

    return simulation

