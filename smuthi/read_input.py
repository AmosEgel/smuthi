# -*- coding: utf-8 -*-
import yaml
import smuthi.simulation
import numpy as np
import smuthi.index_conversion as idx
import smuthi.coordinates as coord
import smuthi.post_processing as pp


def read_input_yaml(filename):
    with open(filename, 'r') as input_file:
        input_data = yaml.load(input_file.read())

    simulation = smuthi.simulation.Simulation()

    simulation.initial_field_collection.vacuum_wavelength = float(input_data['vacuum wavelength'])

    # particle collection
    particle_input = input_data['scattering particles']
    if isinstance(particle_input, str):
        with open(particle_input, 'r') as f:
            first_line = f.readline()
        if first_line[0:-1] == '# spheres':
            particle_data = np.loadtxt(particle_input, skiprows=2)
            for prtcl in particle_data:
                r = prtcl[3]
                n = prtcl[4] + 1j * prtcl[5]
                pos = prtcl[:3]
                simulation.particle_collection.add_sphere(radius=r, refractive_index=n, position=pos)
        else:
            raise ValueError('Currently, only spheres are implemented')
    else:
        for prtcl in input_data['scattering particles']:
            if prtcl['shape'] == 'sphere':
                r = float(prtcl['radius'])
                n = (float(prtcl['refractive index']) + 1j * float(prtcl['extinction coefficient']))
                pos = [float(prtcl['position'][0]), float(prtcl['position'][1]), float(prtcl['position'][2])]
                simulation.particle_collection.add_sphere(radius=r, refractive_index=n, position=pos)
            else:
                raise ValueError('Currently, only spheres are implemented')

    # layer system
    thick = [float(d) for d in input_data['layer system'][0]['thicknesses']]
    ref_ind = [float(n) for n in input_data['layer system'][0]['refractive indices']]
    ext_coeff = [float(n) for n in input_data['layer system'][0]['extinction coefficients']]
    ref_ind = np.array(ref_ind) + 1j * np.array(ext_coeff)
    ref_ind = ref_ind.tolist()
    simulation.layer_system.__init__(thicknesses=thick, refractive_indices=ref_ind)

    # initial field
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

    # linear system
    lmax = int(input_data['lmax'])
    mmax = int(input_data.get('mmax'))
    simulation.linear_system.swe_specs = idx.swe_specifications(lmax, mmax)

    # contour
    neff_waypoints = [complex(nf) for nf in input_data['neff waypoints']]
    neff_discretization = float(input_data['neff discretization'])
    simulation.wr_neff_contour = coord.ComplexContour(neff_waypoints=neff_waypoints,
                                                      neff_discretization=neff_discretization)

    # post processing
    for item in input_data['post processing']:
        if item['task'] == 'plot 2D far-field distribution':
            simulation.post_processing.tasks.append(item)

    return simulation

