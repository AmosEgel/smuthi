# -*- coding: utf-8 -*-
import numpy as np
import smuthi.far_field as ff
import smuthi.near_field as nf
import matplotlib.pyplot as plt
import sys


class PostProcessing:
    def __init__(self):
        self.tasks = []

    def run(self, simulation):
        particle_collection = simulation.particle_collection
        linear_system = simulation.linear_system
        layer_system = simulation.layer_system
        initial_field_collection = simulation.initial_field_collection
        for item in self.tasks:
            if item['task'] == 'evaluate cross sections':
                polar_angles = item.get('polar angles')
                azimuthal_angles = item.get('azimuthal angles')
                layerresponse_precision = item.get('layerresponse precision')
                # filename_forward = item.get('filename forward')
                # filename_backward = item.get('filename backward')

                if (len(initial_field_collection.specs_list) > 1
                    or not initial_field_collection.specs_list[0]['type'] == 'plane wave'):
                    raise ValueError('Cross section only defined for single plane wave excitation.')

                i_top = layer_system.number_of_layers() - 1
                beta_P = initial_field_collection.specs_list[0]['polar angle']
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

                self.scattering_cross_section = ff.scattering_cross_section(
                    polar_angles=polar_angles, initial_field_collection=initial_field_collection,
                    azimuthal_angles=azimuthal_angles, particle_collection=particle_collection,
                    linear_system=linear_system, layer_system=layer_system,
                    layerresponse_precision=layerresponse_precision)

                self.extinction_cross_section = ff.extinction_cross_section(
                    initial_field_collection=initial_field_collection, particle_collection=particle_collection,
                    linear_system=linear_system, layer_system=layer_system,
                    layerresponse_precision=layerresponse_precision)

                # distinguish the cases of top/bottom illumination
                print()
                print('-------------------------------------------------------------------------')
                print('Cross sections:')
                if i_P == 0:
                    print('Scattering into bottom layer (diffuse reflection):  ',
                          self.scattering_cross_section['total bottom'][0]
                          + self.scattering_cross_section['total bottom'][1], ' ' + simulation.length_unit + '^2')

                    if n_transm.imag == 0:
                        print('Scattering into top layer (diffuse transmission):   ',
                              self.scattering_cross_section['total top'][0]
                              + self.scattering_cross_section['total top'][1],
                              ' ' + simulation.length_unit + '^2')

                        print('Total scattering cross section:                     ',
                              self.scattering_cross_section['total'][0] + self.scattering_cross_section['total'][1],
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
                          self.scattering_cross_section['total top'][0] + self.scattering_cross_section['total top'][1],
                          ' ' + simulation.length_unit + '^2')

                    if n_transm.imag == 0:
                        print('Scattering into bottom layer (diffuse transmission):  ',
                              self.scattering_cross_section['total bottom'][0] +
                              self.scattering_cross_section['total bottom'][1],
                              ' ' + simulation.length_unit + '^2')

                        print('Total scattering cross section:                       ',
                              self.scattering_cross_section['total'][0] + self.scattering_cross_section['total'][1],
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

                if item.get('show plots', False):

                    # dsc as polar plot
                    if layer_system.refractive_indices[i_top].imag == 0:
                        # top layer
                        top_idcs = (self.scattering_cross_section['polar angles'] <= np.pi / 2)
                        alpha_grid, beta_grid = np.meshgrid(self.scattering_cross_section['azimuthal angles'],
                                                            self.scattering_cross_section['polar angles'][top_idcs].real
                                                            * 180 / np.pi)

                        fig = plt.figure()
                        ax = fig.add_subplot(111, polar=True)
                        ax.pcolormesh(alpha_grid, beta_grid, (self.scattering_cross_section['differential'][0, top_idcs, :] +
                                                              self.scattering_cross_section['differential'][1, top_idcs, :]))
                        plt.title('DCS in top layer (' + simulation.length_unit + '^2)')

                        plt.figure()
                        plt.plot(self.scattering_cross_section['polar angles'][top_idcs] * 180 / np.pi,
                                 (self.scattering_cross_section['polar'][0, top_idcs]
                                  + self.scattering_cross_section['polar'][1, top_idcs]) * np.pi / 180)
                        plt.xlabel('polar angle (degree)')
                        plt.ylabel('d_CS/d_beta (' + simulation.length_unit + '^2)')
                        plt.title('Polar differential scattering cross section in top layer')
                        plt.grid(True)

                    if layer_system.refractive_indices[0].imag == 0:
                        # bottom layer
                        bottom_idcs = (self.scattering_cross_section['polar angles'] > np.pi / 2)
                        alpha_grid, beta_grid = np.meshgrid(self.scattering_cross_section['azimuthal angles'],
                                                            self.scattering_cross_section['polar angles'][bottom_idcs].real
                                                            * 180 / np.pi)

                        fig = plt.figure()
                        ax = fig.add_subplot(111, polar=True)
                        ax.pcolormesh(alpha_grid, 180 - beta_grid, (self.scattering_cross_section['differential'][0, bottom_idcs, :] +
                                                                      self.scattering_cross_section['differential'][1, bottom_idcs, :]))
                        plt.title('DCS in bottom layer (' + simulation.length_unit + '^2)')

                        plt.figure()
                        plt.plot(180 - self.scattering_cross_section['polar angles'][bottom_idcs] * 180 / np.pi,
                                 (self.scattering_cross_section['polar'][0, bottom_idcs]
                                  + self.scattering_cross_section['polar'][1, bottom_idcs]) * np.pi / 180)
                        plt.xlabel('polar angle (degree)')
                        plt.ylabel('d_CS/d_beta (' + simulation.length_unit + '^2)')
                        plt.title('Polar differential scattering cross section in bottom layer')
                        plt.grid(True)

            elif item['task'] == 'show near field':
                sys.stdout.write("\nEvaluate near fields ... ")
                sys.stdout.flush()

                quantities_to_plot = item['quantities to plot']
                xmin = item.get('xmin', 0)
                xmax = item.get('xmax', 0)
                ymin = item.get('ymin', 0)
                ymax = item.get('ymax', 0)
                zmin = item.get('zmin', 0)
                zmax = item.get('zmax', 0)
                neff_max = item.get('maximal n_effective',
                                    max([n.real for n in simulation.layer_system.refractive_indices]) + 0.5)
                neff_resol = item.get('n_effective resolution', 1e-2)
                n_effective = np.linspace(0, neff_max, neff_max / neff_resol + 2, endpoint=True)
                azimuthal_angles_resol = item.get('azimuthal angles resolution', np.pi / 100)
                azimuthal_angles = np.linspace(0, 2 * np.pi, 2 * np.pi / azimuthal_angles_resol + 1)
                max_field = item.get('maximal field strength')
                max_particle_distance = item.get('maximal particle distance', float('inf'))
                resolution = item.get('spatial resolution', 25)
                interpolate = item.get('interpolation spatial resolution', 5)
                for field_type in item.get('field types', ['scattered']):
                    if field_type == 'scattered':
                        nf.show_scattered_field(quantities_to_plot=quantities_to_plot, xmin=xmin, xmax=xmax, ymin=ymin,
                                                ymax=ymax, zmin=zmin, zmax=zmax, n_effective=n_effective,
                                                azimuthal_angles=azimuthal_angles, length_unit=simulation.length_unit,
                                                vacuum_wavelength=simulation.initial_field_collection.vacuum_wavelength,
                                                particle_collection=simulation.particle_collection,
                                                linear_system=simulation.linear_system, max_field=max_field,
                                                layer_system=simulation.layer_system, resolution=resolution,
                                                max_particle_distance=max_particle_distance, interpolate=interpolate)

                sys.stdout.write("done. \n")
                sys.stdout.flush()
