# -*- coding: utf-8 -*-
import yaml
import smuthi.simulation
import numpy as np
import smuthi.index_conversion as idx
import smuthi.coordinates as coord
import os


def read_input_yaml(filename):
    print('\nReading ' + os.path.abspath(filename))
    with open(filename, 'r') as input_file:
        input_data = yaml.load(input_file.read())

    simulation = smuthi.simulation.Simulation()

    simulation.initial_field_collection.vacuum_wavelength = float(input_data['vacuum wavelength'])
    simulation.length_unit = input_data.get('length unit')

    # particle collection
    particle_input = input_data['scattering particles']
    if isinstance(particle_input, str):
        particle_type = 'sphere'
        with open(particle_input, 'r') as particle_specs_file:
            for line in particle_specs_file:
                if len(line.split()) > 0:
                    if line.split()[-1] == 'spheres':
                        particle_type = 'sphere'
                    elif line.split()[-1] == 'spheroids':
                        particle_type = 'spheroid'
                    elif line.split()[-1] == 'cylinders':
                        particle_type = 'finite cylinder'
                    if not line.split()[0] == '#':
                        numeric_line_data = [float(x) for x in line.split()]
                        pos = numeric_line_data[:3]
                        if particle_type == 'sphere':
                            r = numeric_line_data[3]
                            n = numeric_line_data[4] + 1j * numeric_line_data[5]
                            simulation.particle_collection.add_sphere(radius=r, refractive_index=n, position=pos)
                        if particle_type == 'spheroid':
                            c = numeric_line_data[3]
                            a = numeric_line_data[4]
                            n = numeric_line_data[5] + 1j * numeric_line_data[6]
                            simulation.particle_collection.add_spheroid(semi_axis_c=c, semi_axis_a=a,
                                                                        refractive_index=n, position=pos)
                        if particle_type == 'finite cylinder':
                            r = numeric_line_data[3]
                            h = numeric_line_data[4]
                            n = numeric_line_data[5] + 1j * numeric_line_data[6]
                            simulation.particle_collection.add_finite_cylinder(cylinder_radius=r, cylinder_height=h,
                                                                               refractive_index=n, position=pos)
    else:
        for prtcl in input_data['scattering particles']:
            if prtcl['shape'] == 'sphere':
                r = float(prtcl['radius'])
                n = (float(prtcl['refractive index']) + 1j * float(prtcl['extinction coefficient']))
                pos = [float(prtcl['position'][0]), float(prtcl['position'][1]), float(prtcl['position'][2])]
                simulation.particle_collection.add_sphere(radius=r, refractive_index=n, position=pos)
            elif prtcl['shape'] == 'spheroid':
                c = float(prtcl['semi axis c'])
                a = float(prtcl['semi axis a'])
                n = float(prtcl['refractive index']) + 1j * float(prtcl['extinction coefficient'])
                pos = [float(prtcl['position'][0]), float(prtcl['position'][1]), float(prtcl['position'][2])]
                euler_angles = [float(prtcl['euler angles'][0]), float(prtcl['euler angles'][1]),
                                float(prtcl['euler angles'][2])]
                simulation.particle_collection.add_spheroid(semi_axis_c=c, semi_axis_a=a, refractive_index=n,
                                                            position=pos, euler_angles=euler_angles)
            elif prtcl['shape'] == 'finite cylinder':
                h = float(prtcl['cylinder height'])
                r = float(prtcl['cylinder radius'])
                n = float(prtcl['refractive index']) + 1j * float(prtcl['extinction coefficient'])
                pos = [float(prtcl['position'][0]), float(prtcl['position'][1]), float(prtcl['position'][2])]
                euler_angles = [float(prtcl['euler angles'][0]), float(prtcl['euler angles'][1]),
                                float(prtcl['euler angles'][2])]
                simulation.particle_collection.add_finite_cylinder(cylinder_radius=r, cylinder_height=h,
                                                                   refractive_index=n, position=pos,
                                                                   euler_angles=euler_angles)
            else:
                raise ValueError('Currently, only spheres, spheroids and finite cylinders are implemented')

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
            elif infld['polarization'] == 'TM':
                pol = 1
            else:
                raise ValueError('polarization must be "TE" or "TM"')
            ref = [float(infld['reference point'][0]), float(infld['reference point'][1]),
                   float(infld['reference point'][2])]
            simulation.initial_field_collection.add_planewave(amplitude=a, polar_angle=pol_ang, reference_point=ref,
                                                              azimuthal_angle=az_ang, polarization=pol)

    # linear system
    lmax = int(input_data['lmax'])
    mmax = int(input_data.get('mmax', lmax))
    idx.set_swe_specs(l_max=lmax, m_max=mmax)

    # contour
    neff_waypoints = [complex(nf) for nf in input_data['neff waypoints']]
    neff_discretization = float(input_data['neff discretization'])
    simulation.wr_neff_contour = coord.ComplexContour(neff_waypoints=neff_waypoints,
                                                      neff_discretization=neff_discretization)

    # T-matrix method
    simulation.t_matrix_method = input_data.get('tmatrix method')


    # post processing
    for item in input_data['post processing']:
        if item['task'] == 'evaluate cross sections':
            simulation.post_processing.tasks.append(item)

    return simulation
