# -*- coding: utf-8 -*-
import numpy as np
import smuthi.scattered_field as sf
import smuthi.near_field as nf
import matplotlib.pyplot as plt
import sys
import os


class PostProcessing:
    def __init__(self):
        self.tasks = []
        self.top_scattering_cross_section = None
        self.bottom_scattering_cross_section = None
        self.extinction_cross_section = None

    def run(self, simulation):
        particle_list = simulation.particle_list
        layer_system = simulation.layer_system
        initial_field = simulation.initial_field
        for item in self.tasks:
            if item['task'] == 'evaluate cross sections':
                polar_angles = item.get('polar angles')
                azimuthal_angles = item.get('azimuthal angles')

                self.top_scattering_cross_section, self.bottom_scattering_cross_section = sf.scattering_cross_section(
                    initial_field=initial_field, polar_angles=polar_angles, azimuthal_angles=azimuthal_angles,
                    particle_list=particle_list, layer_system=layer_system)

                outputdir = simulation.output_dir + '/far_field'
                if (not os.path.exists(outputdir)) and (item.get('save data', False) or item.get('save plots', False)):
                    os.makedirs(outputdir)

                if item.get('save data', False):
                    dsc_top, dsc_bottom = self.top_scattering_cross_section, self.bottom_scattering_cross_section
                    if dsc_top:
                        np.savetxt(outputdir + '/top_differential_cross_section_TE.dat', dsc_top.signal[0, :, :],
                                   header='Differential TE polarized cross section into top hemisphere. Each line'
                                   ' corresponds to a polar angle, each column corresponds to an azimuthal angle.')
                        np.savetxt(outputdir + '/top_differential_cross_section_TM.dat', dsc_top.signal[1, :, :],
                                   header='Differential TM polarized cross section into top hemisphere. Each line'
                                   ' corresponds to a polar angle, each column corresponds to an azimuthal angle.')
                        np.savetxt(outputdir + '/top_polar_differential_cross_section_TE.dat', 
                                   dsc_top.azimuthal_integral()[0, :], header='Polar differential TE polarized cross ' 
                                   'section into top hemisphere. Each line corresponds to a polar angle.')
                        np.savetxt(outputdir + '/top_polar_differential_cross_section_TM.dat', 
                                   dsc_top.azimuthal_integral()[1, :], header='Polar differential TM polarized cross ' 
                                   'section into top hemisphere. Each line corresponds to a polar angle.')
                        np.savetxt(outputdir + '/top_polar_angles.dat', dsc_top.polar_angles, 
                                   header= 'Polar angles of the far field in the top hemisphere in radians.')
                    if dsc_bottom:
                        np.savetxt(outputdir + '/bottom_differential_cross_section_TE.dat', dsc_bottom.signal[0, :, :],
                                   header='Differential TE polarized cross section into bottom hemisphere. Each line'
                                   ' corresponds to a polar angle, each column corresponds to an azimuthal angle.')
                        np.savetxt(outputdir + '/bottom_differential_cross_section_TM.dat', dsc_bottom.signal[1, :, :],
                                   header='Differential TM polarized cross section into bottom hemisphere. Each line'
                                   ' corresponds to a polar angle, each column corresponds to an azimuthal angle.')
                        np.savetxt(outputdir + '/bottom_polar_differential_cross_section_TE.dat', 
                                   dsc_bottom.azimuthal_integral()[0, :], header='Polar differential TE polarized '
                                   'cross section into bottom hemisphere. Each line corresponds to a polar angle.')
                        np.savetxt(outputdir + '/bottom_polar_differential_cross_section_TM.dat', 
                                   dsc_bottom.azimuthal_integral()[1, :], header='Polar differential TM polarized '
                                   'cross section into bottom hemisphere. Each line corresponds to a polar angle.')
                        np.savetxt(outputdir + '/bottom_polar_angles.dat', dsc_bottom.polar_angles, 
                                   header= 'Polar angles of the far field in the bottom hemisphere in radians.')
                    np.savetxt(outputdir + '/azimuthal_angles.dat', self.azimuthal_angles, 
                               header='Azimuthal angles of the far field in radians.')

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
                          (self.bottom_scattering_cross_section.integral()[0] 
                          + self.bottom_scattering_cross_section.integral()[1]).real, ' ' + simulation.length_unit 
                          + '^2')
                    tot_scat = (self.bottom_scattering_cross_section.integral()[0] 
                                + self.bottom_scattering_cross_section.integral()[1]).real
                    if n_transm.imag == 0:
                        print('Scattering into top layer (diffuse transmission):  ',
                              (self.top_scattering_cross_section.integral()[0]
                              + self.top_scattering_cross_section.integral()[1]).real, ' ' + simulation.length_unit 
                              + '^2')
                        tot_scat += (self.top_scattering_cross_section.integral()[0] 
                                     + self.top_scattering_cross_section.integral()[1]).real
                        
                        print('Total scattering cross section:                     ',
                              tot_scat, ' ' + simulation.length_unit + '^2')

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
                          (self.top_scattering_cross_section.integral()[0]
                           + self.top_scattering_cross_section.integral()[1]).real,
                          ' ' + simulation.length_unit + '^2')
                    if n_transm.imag == 0:
                        print('Scattering into bottom layer (diffuse transmission):  ',
                              (self.bottom_scattering_cross_section.integral()[0] +
                               self.bottom_scattering_cross_section.integral()[1]).real,
                              ' ' + simulation.length_unit + '^2')
                        print('Total scattering cross section:                       ',
                              (self.top_scattering_cross_section.integral()[0]
                               + self.top_scattering_cross_section.integral()[1]
                               + self.bottom_scattering_cross_section.integral()[0]
                               + self.bottom_scattering_cross_section.integral()[1]).real,
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

                # plot the far field

                # dsc as polar plot
                if self.top_scattering_cross_section:
                    # top layer
                    alpha_grid, beta_grid = np.meshgrid(self.top_scattering_cross_section.azimuthal_angles,
                                                        self.top_scattering_cross_section.polar_angles.real 
                                                        * 180 / np.pi)

                    fig = plt.figure()
                    ax = fig.add_subplot(111, polar=True)
                    pcmsh = ax.pcolormesh(alpha_grid, beta_grid,
                                  (self.top_scattering_cross_section.signal[0, :, :]
                                   + self.top_scattering_cross_section.signal[1, :, :]), cmap='inferno')
                    plt.colorbar(pcmsh, ax=ax)
                    plt.title('DCS in top layer (' + simulation.length_unit + '^2)')

                    if item.get('save plots', False):
                        plt.savefig(outputdir + '/top_dcs.png')
                    if item.get('show plots', False):
                        plt.draw()
                    else:
                        plt.close(fig)
                    
                    fig = plt.figure()
                    plt.plot(self.top_scattering_cross_section.polar_angles * 180 / np.pi,
                             np.sum(self.top_scattering_cross_section.azimuthal_integral(), axis=0) * np.pi / 180)
                    plt.xlabel('polar angle (degree)')
                    plt.ylabel('d_CS/d_beta (' + simulation.length_unit + '^2/deg)')
                    plt.title('Polar differential scattering cross section in top layer')
                    plt.grid(True)

                    if item.get('save plots', False):
                        plt.savefig(outputdir + '/top_polar_dcs.png')
                    if item.get('show plots', False):
                        plt.draw()
                    else:
                        plt.close(fig)
                if self.bottom_scattering_cross_section:
                    # bottom layer
                    alpha_grid, beta_grid = np.meshgrid(self.bottom_scattering_cross_section.azimuthal_angles,
                                                        self.bottom_scattering_cross_section.polar_angles.real 
                                                        * 180 / np.pi)

                    fig = plt.figure()
                    ax = fig.add_subplot(111, polar=True)
                    pcmsh = ax.pcolormesh(alpha_grid, beta_grid,
                                  (self.bottom_scattering_cross_section.signal[0, :, :]
                                   + self.bottom_scattering_cross_section.signal[1, :, :]), cmap='inferno')
                    plt.colorbar(pcmsh, ax=ax)
                    plt.title('DCS in bottom layer (' + simulation.length_unit + '^2)')

                    if item.get('save plots', False):
                        plt.savefig(outputdir + '/bottom_dcs.png')
                    if item.get('show plots', False):
                        plt.draw()
                    else:
                        plt.close(fig)

                    fig = plt.figure()
                    plt.plot(self.bottom_scattering_cross_section.polar_angles * 180 / np.pi,
                             np.sum(self.bottom_scattering_cross_section.azimuthal_integral(), axis=0) * np.pi / 180)
                    plt.xlabel('polar angle (degree)')
                    plt.ylabel('d_CS/d_beta (' + simulation.length_unit + '^2/deg)')
                    plt.title('Polar differential scattering cross section in bottom layer')
                    plt.grid(True)

                    if item.get('save plots', False):
                        plt.savefig(outputdir + '/bottom_polar_dcs.png')
                    if item.get('show plots', False):
                        plt.draw()
                    else:
                        plt.close(fig)
            elif item['task'] == 'evaluate near field':
                sys.stdout.write("\nEvaluate near fields ... ")
                sys.stdout.flush()

                quantities_to_plot = item['quantities to plot']

                show_plots = item.get('show plots', False)
                save_plots = item.get('save plots', False)
                save_animations = item.get('save animations', False)
                save_data = item.get('save data', False)

                if simulation.output_dir:
                    outputdir = simulation.output_dir + '/near_field'
                else:
                    outputdir = '.'

                if (not os.path.exists(outputdir)) and (save_plots or save_animations or save_data):
                    os.makedirs(outputdir)


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
                k_parallel = n_effective * simulation.initial_field.angular_frequency()
                azimuthal_angles_resol = item.get('azimuthal angles resolution', np.pi / 100)
                azimuthal_angles = np.linspace(0, 2 * np.pi, 2 * np.pi / azimuthal_angles_resol + 1)
                max_field = item.get('maximal field strength')
                max_particle_distance = item.get('maximal particle distance', float('inf'))
                resolution = item.get('spatial resolution', 25)
                interpolate = item.get('interpolation spatial resolution', 5)
                nf.show_near_field(quantities_to_plot=quantities_to_plot, show_plots=show_plots, save_plots=save_plots,
                                   save_animations=save_animations, save_data=save_data, outputdir=outputdir,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                                   k_parallel=k_parallel, azimuthal_angles=azimuthal_angles, simulation=simulation,
                                   max_field=max_field, resolution=resolution,
                                   max_particle_distance=max_particle_distance, interpolate=interpolate)

                sys.stdout.write("done. \n")
                sys.stdout.flush()
