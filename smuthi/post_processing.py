# -*- coding: utf-8 -*-
import numpy as np
import smuthi.scattered_field as sf
import smuthi.graphical_output as go
import sys
import os


class PostProcessing:
    def __init__(self):
        self.tasks = []
        self.scattering_cross_section = None
        self.extinction_cross_section = None

    def run(self, simulation):
        particle_list = simulation.particle_list
        layer_system = simulation.layer_system
        initial_field = simulation.initial_field
        for item in self.tasks:
            if item['task'] == 'evaluate cross sections':
                polar_angles = item.get('polar angles')
                azimuthal_angles = item.get('azimuthal angles')

                self.scattering_cross_section = sf.scattering_cross_section(
                    initial_field=initial_field, polar_angles=polar_angles, azimuthal_angles=azimuthal_angles,
                    particle_list=particle_list, layer_system=layer_system)

                outputdir = simulation.output_dir + '/far_field'
                if (not os.path.exists(outputdir)) and (item.get('save data', False) or item.get('save plots', False)):
                    os.makedirs(outputdir)

                if item.get('save data', False):
                    self.scattering_cross_section.export(output_directory=outputdir, tag='dsc')

                # extinction_cross_section(initial_field, particle_list, layer_system)
                self.extinction_cross_section = sf.extinction_cross_section(initial_field, particle_list, layer_system)

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
                          (self.scattering_cross_section.bottom().integral()[0]
                          + self.scattering_cross_section.bottom().integral()[1]).real, ' ' + simulation.length_unit
                          + '^2')
                    if n_transm.imag == 0:
                        print('Scattering into top layer (diffuse transmission):  ',
                              (self.scattering_cross_section.top().integral()[0]
                              + self.scattering_cross_section.top().integral()[1]).real, ' ' + simulation.length_unit
                              + '^2')
                        print('Total scattering cross section:                       ',
                              (self.scattering_cross_section.top().integral()[0]
                               + self.scattering_cross_section.top().integral()[1]
                               + self.scattering_cross_section.bottom().integral()[0]
                               + self.scattering_cross_section.bottom().integral()[1]).real,
                              ' ' + simulation.length_unit + '^2')
                    print('Bottom layer extinction (extinction of reflection): ',
                          self.extinction_cross_section['bottom'].real,
                          ' ' + simulation.length_unit + '^2')
                    if n_transm.imag == 0:
                        print('Top layer extinction (extinction of transmission):  ',
                              self.extinction_cross_section['top'].real,
                              ' ' + simulation.length_unit + '^2')
                        print('Total extinction cross section:                     ',
                              (self.extinction_cross_section['top'] + self.extinction_cross_section['bottom']).real,
                              ' ' + simulation.length_unit + '^2')
                else:
                    print('Scattering into top layer (diffuse reflection):       ',
                          (self.scattering_cross_section.top().integral()[0]
                           + self.scattering_cross_section.top().integral()[1]).real,
                          ' ' + simulation.length_unit + '^2')
                    if n_transm.imag == 0:
                        print('Scattering into bottom layer (diffuse transmission):  ',
                              (self.scattering_cross_section.bottom().integral()[0] +
                               self.scattering_cross_section.bottom().integral()[1]).real,
                              ' ' + simulation.length_unit + '^2')
                        print('Total scattering cross section:                       ',
                              (self.scattering_cross_section.top().integral()[0]
                               + self.scattering_cross_section.top().integral()[1]
                               + self.scattering_cross_section.bottom().integral()[0]
                               + self.scattering_cross_section.bottom().integral()[1]).real,
                              ' ' + simulation.length_unit + '^2')

                    print('Top layer extinction (extinction of reflection):      ',
                          self.extinction_cross_section['top'],
                          ' ' + simulation.length_unit + '^2')

                    if n_transm.imag == 0:
                        print('Bottom layer extinction (extinction of transmission): ',
                              self.extinction_cross_section['bottom'],
                              ' ' + simulation.length_unit + '^2')

                        print('Total extinction cross section:                       ',
                              self.extinction_cross_section['top'] + self.extinction_cross_section['bottom'],
                              ' ' + simulation.length_unit + '^2')
                print('-------------------------------------------------------------------------')

                # plot dsc
                if any(self.scattering_cross_section.polar_angles <= np.pi/2):
                    go.show_far_field(far_field=self.scattering_cross_section.top(),
                                      save_plots=item.get('save plots', False),
                                      show_plots=item.get('show plots', False),
                                      tag='top_dsc', outputdir=outputdir)

                if any(self.scattering_cross_section.polar_angles >= np.pi/2):
                    go.show_far_field(far_field=self.scattering_cross_section.bottom(),
                                      save_plots=item.get('save plots', False),
                                      show_plots=item.get('show plots', False),
                                      tag='bottom_dsc', outputdir=outputdir)

            elif item['task'] == 'evaluate near field':
                sys.stdout.write("\nEvaluate near fields ... ")
                sys.stdout.flush()

                quantities_to_plot = item['quantities to plot']

                if simulation.output_dir:
                    outputdir = simulation.output_dir + '/near_field'
                else:
                    outputdir = '.'

                if (not os.path.exists(outputdir)) \
                        and (item.get('save plots', False) or item.get('save animations', False)):
                    os.makedirs(outputdir)

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
