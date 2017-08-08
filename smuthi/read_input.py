# -*- coding: utf-8 -*-
import yaml
import smuthi.simulation
import numpy as np
import smuthi.coordinates as coord
import smuthi.particles as part
import smuthi.initial_field as init
import smuthi.layers as lay
import smuthi.post_processing as pp
import os


def read_input_yaml(filename):
    print('\nReading ' + os.path.abspath(filename))
    with open(filename, 'r') as input_file:
        input_data = yaml.load(input_file.read())

    simulation = smuthi.simulation.Simulation(input_file=filename, output_dir=input_data.get('output folder'),
                                              save_after_run=input_data.get('save simulation'))

    simulation.length_unit = input_data.get('length unit')

    # particle collection
    particle_list = []
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
                            l_max = numeric_line_data[6]
                            m_max = numeric_line_data[7]
                            particle_list.append(part.Sphere(position=pos, refractive_index=n, radius=r, l_max=l_max,
                                                             m_max=m_max))
                        if particle_type == 'spheroid':
                            c = numeric_line_data[3]
                            a = numeric_line_data[4]
                            n = numeric_line_data[5] + 1j * numeric_line_data[6]
                            l_max = numeric_line_data[7]
                            m_max = numeric_line_data[8]
                            particle_list.append(part.Spheroid(position=pos, refractive_index=n, semi_axis_c=c,
                                                               semi_axis_a=a, l_max=l_max, m_max=m_max))
                        if particle_type == 'finite cylinder':
                            r = numeric_line_data[3]
                            h = numeric_line_data[4]
                            n = numeric_line_data[5] + 1j * numeric_line_data[6]
                            l_max = numeric_line_data[7]
                            m_max = numeric_line_data[8]
                            particle_list.append(part.FiniteCylinder(position=pos, refractive_index=n,
                                                                     cylinder_radius=r, cylinder_height=h, l_max=l_max,
                                                                     m_max=m_max))
    else:
        for prtcl in input_data['scattering particles']:
            n = (float(prtcl['refractive index']) + 1j * float(prtcl['extinction coefficient']))
            pos = [float(prtcl['position'][0]), float(prtcl['position'][1]), float(prtcl['position'][2])]
            l_max = int(prtcl['l_max'])
            m_max = int(prtcl['m_max'])
            if prtcl['shape'] == 'sphere':
                r = float(prtcl['radius'])
                particle_list.append(part.Sphere(position=pos, refractive_index=n, radius=r, l_max=l_max, m_max=m_max))
            elif prtcl['shape'] == 'spheroid':
                c = float(prtcl['semi axis c'])
                a = float(prtcl['semi axis a'])
                euler_angles = [float(prtcl['euler angles'][0]), float(prtcl['euler angles'][1]),
                                float(prtcl['euler angles'][2])]
                use_ds = prtcl.get('use discrete sources', True)
                nint = prtcl.get('nint', 200)
                nrank = prtcl.get('nrank', l_max + 2)
                t_matrix_method = {'use discrete sources': use_ds, 'nint': nint, 'nrank': nrank}
                particle_list.append(part.Spheroid(position=pos, refractive_index=n, semi_axis_a=a, semi_axis_c=c,
                                                   l_max=l_max, m_max=m_max, euler_angles=euler_angles,
                                                   t_matrix_method=t_matrix_method))
            elif prtcl['shape'] == 'finite cylinder':
                h = float(prtcl['cylinder height'])
                r = float(prtcl['cylinder radius'])
                euler_angles = [float(prtcl['euler angles'][0]), float(prtcl['euler angles'][1]),
                                float(prtcl['euler angles'][2])]
                use_ds = prtcl.get('use discrete sources', True)
                nint = prtcl.get('nint', 200)
                nrank = prtcl.get('nrank', l_max + 2)
                t_matrix_method = {'use discrete sources': use_ds, 'nint': nint, 'nrank': nrank}

                particle_list.append(part.FiniteCylinder(position=pos, refractive_index=n, cylinder_radius=r,
                                                         cylinder_height=h, l_max=l_max, m_max=m_max,
                                                         euler_angles=euler_angles, t_matrix_method=t_matrix_method))
            else:
                raise ValueError('Currently, only spheres, spheroids and finite cylinders are implemented')
    simulation.particle_list = particle_list

    # layer system
    thick = [float(d) for d in input_data['layer system']['thicknesses']]
    ref_ind = [float(n) for n in input_data['layer system']['refractive indices']]
    ext_coeff = [float(n) for n in input_data['layer system']['extinction coefficients']]
    ref_ind = np.array(ref_ind) + 1j * np.array(ext_coeff)
    ref_ind = ref_ind.tolist()
    simulation.layer_system = lay.LayerSystem(thicknesses=thick, refractive_indices=ref_ind)

    # initial field
    wl = float(input_data['vacuum wavelength'])
    infld = input_data['initial field']
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
        initial_field = init.PlaneWave(vacuum_wavelength=wl, polar_angle=pol_ang, azimuthal_angle=az_ang,
                                       polarization=pol, amplitude=a, reference_point=ref)
    simulation.initial_field = initial_field

    # contour
    neff_waypoints = [complex(nf) for nf in input_data['neff waypoints']]
    neff_discretization = float(input_data['neff discretization'])
    simulation.wr_neff_contour = coord.ComplexContour(neff_waypoints=neff_waypoints,
                                                      neff_discretization=neff_discretization)

    # post processing
    simulation.post_processing = pp.PostProcessing()
    if input_data.get('post processing'):
        for item in input_data['post processing']:
            if item['task'] == 'evaluate cross sections':
                simulation.post_processing.tasks.append(item)
            elif item['task'] == 'evaluate near field':
                simulation.post_processing.tasks.append(item)

    return simulation
