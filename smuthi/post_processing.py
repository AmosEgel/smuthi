# -*- coding: utf-8 -*-
import numpy as np
import smuthi.scattered_field as sf
import smuthi.graphical_output as go
import sys


class PostProcessing:
    def __init__(self):
        self.tasks = []

    def run(self, simulation):
        particle_list = simulation.particle_list
        layer_system = simulation.layer_system
        initial_field = simulation.initial_field
        for item in self.tasks:
            if item['task'] == 'evaluate far field':
                outputdir = simulation.output_dir + '/far_field'
                if item.get('angle units', 'polar') == 'degree':
                    ang_fac = np.pi / 180
                else:
                    ang_fac = 1
                ang_res = item.get('angular resolution', np.pi / 180 / ang_fac) * ang_fac
                polar_angles = np.concatenate([np.arange(0, np.pi/2, ang_res),   # exclude pi/2 as it is singular
                                               np.arange(np.pi/2 + ang_res, np.pi, ang_res), [np.pi]])
                azimuthal_angles = np.concatenate([np.arange(0, 2 * np.pi, ang_res), [2 * np.pi]])
                
                if type(initial_field).__name__ == 'PlaneWave':
                    self.scattering_cross_section, self.extinction_cross_section = evaluate_cross_section(
                        polar_angles, azimuthal_angles, initial_field, particle_list, layer_system, outputdir,
                        show_plots=item.get('show plots', False), save_plots=item.get('save plots', False),
                        save_data=item.get('save data', False), length_unit=simulation.length_unit)
                elif type(initial_field).__name__ == 'GaussianBeam':
                    self.total_far_field, self.initial_far_field, self.scattered_far_field = sf.total_far_field(
                        initial_field=initial_field, particle_list=particle_list, layer_system=layer_system, 
                        polar_angles=polar_angles, azimuthal_angles=azimuthal_angles)    
                
                    go.show_far_field(far_field=self.total_far_field, save_plots=item.get('save plots', False),
                                      show_plots=item.get('show plots', False), save_data=item.get('save data', False),
                                      tag='total_far_field', outputdir=outputdir, flip_downward=True, split=True)
                    go.show_far_field(far_field=self.initial_far_field, save_plots=item.get('save plots', False),
                                      show_plots=item.get('show plots', False), save_data=item.get('save data', False),
                                      tag='initial_far_field', outputdir=outputdir, flip_downward=True, split=True)
                    go.show_far_field(far_field=self.scattered_far_field, save_plots=item.get('save plots', False),
                                      show_plots=item.get('show plots', False), save_data=item.get('save data', False),
                                      tag='scattered_far_field', outputdir=outputdir, flip_downward=True, split=True)
                    
                    in_pow = sum(initial_field.initial_intensity(layer_system).integral()).real
                    top_pow = sum(self.total_far_field.top().integral()).real
                    bottom_pow = sum(self.total_far_field.bottom().integral()).real
                    
                    print()
                    print('-------------------------------------------------------------------------')
                    print('Far field:')
                    print('Initial power:                                         ', in_pow)
                    if initial_field.polar_angle < np.pi / 2:
                        print('Radiation into bottom layer (total reflection):        ', bottom_pow, 
                              ' or ', round(bottom_pow / in_pow * 100, 2), '%')
                        print('Radiation into top layer (total transmission):         ', top_pow,
                              ' or ', round(top_pow / in_pow * 100, 2), '%') 
                    else:
                        print('Radiation into bottom layer (total transmission):      ', bottom_pow, 
                              ' or ', round(bottom_pow / in_pow * 100, 2), '%')
                        print('Radiation into top layer (total reflection):           ', top_pow,
                              ' or ', round(top_pow / in_pow * 100, 2), '%')
                    print('Absorption and incoupling into waveguide modes:        ', in_pow - top_pow - bottom_pow,
                          ' or ', round((in_pow - top_pow - bottom_pow) / in_pow * 100, 2), '%')
                    print('-------------------------------------------------------------------------')
                    
            elif item['task'] == 'evaluate near field':
                sys.stdout.write("\nEvaluate near fields ... ")
                sys.stdout.flush()

                quantities_to_plot = item['quantities to plot']

                if simulation.output_dir:
                    outputdir = simulation.output_dir + '/near_field'
                else:
                    outputdir = '.'

                neff_max = item.get('maximal n_effective',
                                    max([n.real for n in simulation.layer_system.refractive_indices]) + 0.5)
                neff_resol = item.get('n_effective resolution', 1e-2)
                n_effective = np.linspace(0, neff_max, neff_max / neff_resol + 2, endpoint=True)
                k_parallel = n_effective * simulation.initial_field.angular_frequency()
                azimuthal_angles_resol = item.get('azimuthal angles resolution', np.pi / 100)
                azimuthal_angles = np.linspace(0, 2 * np.pi, 2 * np.pi / azimuthal_angles_resol + 1)
                go.show_near_field(quantities_to_plot=quantities_to_plot, show_plots=item.get('show plots', False),
                                   save_plots=item.get('save plots', False), save_data=item.get('save data', False),
                                   save_animations=item.get('save animations', False), outputdir=outputdir,
                                   xmin=item.get('xmin', 0), xmax=item.get('xmax', 0), ymin=item.get('ymin', 0),
                                   ymax=item.get('ymax', 0), zmin=item.get('zmin', 0), zmax=item.get('zmax', 0),
                                   k_parallel=k_parallel, azimuthal_angles=azimuthal_angles, simulation=simulation,
                                   max_field=item.get('maximal field strength'),
                                   resolution=item.get('spatial resolution', 25),
                                   max_particle_distance=item.get('maximal particle distance', float('inf')),
                                   interpolate=item.get('interpolation spatial resolution', 5))

                sys.stdout.write("done. \n")
                sys.stdout.flush()


def evaluate_cross_section(polar_angles, azimuthal_angles, initial_field, particle_list, layer_system, outputdir, 
                           show_plots, save_plots, save_data, length_unit):
    scattering_cross_section = sf.scattering_cross_section(initial_field=initial_field, polar_angles=polar_angles, 
                                                           azimuthal_angles=azimuthal_angles, 
                                                           particle_list=particle_list, layer_system=layer_system)

    if save_data:
        scattering_cross_section.export(output_directory=outputdir, tag='dsc')

    extinction_cross_section = sf.extinction_cross_section(initial_field, particle_list, layer_system)

    # distinguish the cases of top/bottom illumination
    i_top = layer_system.number_of_layers() - 1
    beta_P = initial_field.polar_angle
    if beta_P < np.pi / 2:
        i_P = 0
        n_P = layer_system.refractive_indices[i_P]
        n_transm = layer_system.refractive_indices[i_top]
    else:
        i_P = i_top
        n_P = layer_system.refractive_indices[i_P]
        n_transm = layer_system.refractive_indices[0]
    if n_P.imag:
        raise ValueError('plane wave from absorbing layer: cross section undefined')

    print()
    print('-------------------------------------------------------------------------')
    print('Cross sections:')
    if i_P == 0:
        print('Scattering into bottom layer (diffuse reflection):  ',
              (scattering_cross_section.bottom().integral()[0]
              + scattering_cross_section.bottom().integral()[1]).real, ' ' + length_unit
              + '^2')
        if n_transm.imag == 0:
            print('Scattering into top layer (diffuse transmission):   ',
                  (scattering_cross_section.top().integral()[0]
                  + scattering_cross_section.top().integral()[1]).real, ' ' + length_unit
                  + '^2')
            print('Total scattering cross section:                     ',
                  (scattering_cross_section.top().integral()[0]
                   + scattering_cross_section.top().integral()[1]
                   + scattering_cross_section.bottom().integral()[0]
                   + scattering_cross_section.bottom().integral()[1]).real,
                  ' ' + length_unit + '^2')
        print('Bottom layer extinction (extinction of reflection): ',
              extinction_cross_section['bottom'].real,
              ' ' + length_unit + '^2')
        if n_transm.imag == 0:
            print('Top layer extinction (extinction of transmission):  ',
                  extinction_cross_section['top'].real,
                  ' ' + length_unit + '^2')
            print('Total extinction cross section:                     ',
                  (extinction_cross_section['top'] + extinction_cross_section['bottom']).real,
                  ' ' + length_unit + '^2')
    else:
        print('Scattering into top layer (diffuse reflection):       ',
              (scattering_cross_section.top().integral()[0]
               + scattering_cross_section.top().integral()[1]).real,
              ' ' + length_unit + '^2')
        if n_transm.imag == 0:
            print('Scattering into bottom layer (diffuse transmission):  ',
                  (scattering_cross_section.bottom().integral()[0] +
                   scattering_cross_section.bottom().integral()[1]).real,
                  ' ' + length_unit + '^2')
            print('Total scattering cross section:                       ',
                  (scattering_cross_section.top().integral()[0]
                   + scattering_cross_section.top().integral()[1]
                   + scattering_cross_section.bottom().integral()[0]
                   + scattering_cross_section.bottom().integral()[1]).real,
                  ' ' + length_unit + '^2')

        print('Top layer extinction (extinction of reflection):      ',
              extinction_cross_section['top'],
              ' ' + length_unit + '^2')

        if n_transm.imag == 0:
            print('Bottom layer extinction (extinction of transmission): ',
                  extinction_cross_section['bottom'],
                  ' ' + length_unit + '^2')

            print('Total extinction cross section:                       ',
                  extinction_cross_section['top'] + extinction_cross_section['bottom'],
                  ' ' + length_unit + '^2')
    print('-------------------------------------------------------------------------')

    # plot dsc
    go.show_far_field(far_field=scattering_cross_section, save_plots=save_plots, show_plots=show_plots, tag='dsc', 
                      outputdir=outputdir, flip_downward=True, split=True)
        
    return scattering_cross_section, extinction_cross_section
