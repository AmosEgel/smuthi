# -*- coding: utf-8 -*-
import yaml
import smuthi.simulation
import numpy as np
import smuthi.coordinates as coord
import smuthi.particles as part
import smuthi.initial_field as init
import smuthi.layers as lay
import smuthi.cuda_sources as cu
import smuthi.post_processing as pp
import os


def read_input_yaml(filename):
    """Parse input file
    
    Args:
        filename (str):    relative path and filename of input file
        
    Returns:
        smuthi.simulation.Simulation object containing the params of the input file
    """
    print('\nReading ' + os.path.abspath(filename))
    with open(filename, 'r') as input_file:
        input_data = yaml.load(input_file.read())

    cu.enable_gpu(input_data.get('enable GPU', False))

    # wavelength
    wl = float(input_data['vacuum wavelength'])
    
    # set default coordinate arrays
    angle_unit = input_data.get('angle unit')
    if angle_unit == 'degree':
        angle_factor = np.pi / 180
    else:
        angle_factor = 1
    angle_resolution = input_data.get('angular resolution', np.pi / 180 / angle_factor) * angle_factor
    coord.default_azimuthal_angles = np.arange(0, 2 * np.pi + angle_resolution / 2, angle_resolution)
    coord.default_polar_angles = np.arange(0, np.pi + angle_resolution / 2, angle_resolution)
    
    neff_resolution = input_data.get('neff resolution', 1e-2)
    neff_max = input_data.get('neff max')
    if neff_max is None:
        ref_ind = [float(n) for n in input_data['layer system']['refractive indices']]
        neff_max = max(np.array(ref_ind).real) + 1
    neff_imag = input_data.get('neff imaginary deflection', 5e-2)
    coord.set_default_k_parallel(vacuum_wavelength=wl, neff_resolution=neff_resolution, neff_max=neff_max, 
                                 neff_imag=neff_imag)
    
    # initialize simulation
    lookup_resolution = input_data.get('coupling matrix lookup resolution', None)
    if lookup_resolution <= 0:
        lookup_resolution = None

    simulation = smuthi.simulation.Simulation(solver_type=input_data.get('solver type', 'LU'),
                                              solver_tolerance=float(input_data.get('solver tolerance', 1e-4)),
                                              store_coupling_matrix=input_data.get('store coupling matrix', True),
                                              coupling_matrix_lookup_resolution=lookup_resolution,
                                              coupling_matrix_interpolator_kind=input_data.get('interpolation order',
                                                                                               'linear'),
                                              input_file=filename,
                                              length_unit=input_data.get('length unit'),
                                              output_dir=input_data.get('output folder'), 
                                              save_after_run=input_data.get('save simulation'))

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
                nfmds_settings = prtcl.get('NFM-DS settings', {})
                use_ds = nfmds_settings.get('use discrete sources', True)
                nint = nfmds_settings.get('nint', 200)
                nrank = nfmds_settings.get('nrank', l_max + 2)
                t_matrix_method = {'use discrete sources': use_ds, 'nint': nint, 'nrank': nrank}
                particle_list.append(part.Spheroid(position=pos, refractive_index=n, semi_axis_a=a, semi_axis_c=c,
                                                   l_max=l_max, m_max=m_max, euler_angles=euler_angles,
                                                   t_matrix_method=t_matrix_method))
            elif prtcl['shape'] == 'finite cylinder':
                h = float(prtcl['cylinder height'])
                r = float(prtcl['cylinder radius'])
                euler_angles = [float(prtcl['euler angles'][0]), float(prtcl['euler angles'][1]),
                                float(prtcl['euler angles'][2])]
                nfmds_settings = prtcl.get('NFM-DS settings', {})
                use_ds = nfmds_settings.get('use discrete sources', True)
                nint = nfmds_settings.get('nint', 200)
                nrank = nfmds_settings.get('nrank', l_max + 2)
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
    infld = input_data['initial field']
    if infld['type'] == 'plane wave':
        a = float(infld['amplitude'])
        pol_ang = angle_factor * float(infld['polar angle'])
        az_ang = angle_factor * float(infld['azimuthal angle'])
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
    elif infld['type'] == 'Gaussian beam':
        a = float(infld['amplitude'])
        pol_ang = angle_factor * float(infld['polar angle'])
        az_ang = angle_factor * float(infld['azimuthal angle'])
        if infld['polarization'] == 'TE':
            pol = 0
        elif infld['polarization'] == 'TM':
            pol = 1
        else:
            raise ValueError('polarization must be "TE" or "TM"')
        ref = [float(infld['focus point'][0]), float(infld['focus point'][1]), float(infld['focus point'][2])]
        ang_res = infld.get('angular resolution', np.pi / 180 / ang_fac) * ang_fac
        bet_arr = np.arange(0, np.pi/2, ang_res)
        if pol_ang <= np.pi:
            kparr = np.sin(bet_arr) * simulation.layer_system.wavenumber(layer_number=0, vacuum_wavelength=wl)
        else:
            kparr = np.sin(bet_arr) * simulation.layer_system.wavenumber(layer_number=-1, vacuum_wavelength=wl)
        wst = infld['beam waist']
        aarr = np.concatenate([np.arange(0, 2 * np.pi, ang_res), [2 * np.pi]])
        initial_field = init.GaussianBeam(vacuum_wavelength=wl, polar_angle=pol_ang, azimuthal_angle=az_ang,
                                          polarization=pol, beam_waist=wst, k_parallel_array=kparr,
                                          azimuthal_angles_array=aarr, amplitude=a, reference_point=ref)
    elif infld['type'] == 'dipole source':
        pos = [float(infld['position'][i]) for i in range(3)] 
        mom = [float(infld['dipole moment'][i]) for i in range(3)]
        initial_field = init.DipoleSource(vacuum_wavelength=wl, dipole_moment=mom, position=pos)
    elif infld['type'] == 'dipole collection':
        initial_field = init.DipoleCollection(vacuum_wavelength=wl)
        dipoles = infld['dipoles']
        for dipole in dipoles:
            pos = [float(dipole['position'][i]) for i in range(3)] 
            mom = [float(dipole['dipole moment'][i]) for i in range(3)]
            dip = init.DipoleSource(vacuum_wavelength=wl, dipole_moment=mom, position=pos)
            initial_field.append(dip)
    simulation.initial_field = initial_field

    # post processing
    simulation.post_processing = pp.PostProcessing()
    if input_data.get('post processing'):
        for item in input_data['post processing']:
            if item['task'] == 'evaluate far field':
                simulation.post_processing.tasks.append(item)
            elif item['task'] == 'evaluate near field':
                simulation.post_processing.tasks.append(item)

    return simulation
