# -*- coding: utf-8 -*-
"""Manage all post processing tasks after solving the linear system."""
import numpy as np
import smuthi.scattered_field as sf
import smuthi.graphical_output as go
import sys


class PostProcessing:
    def __init__(self):
        self.tasks = []

    def run(self, simulation):
        """Run tasks for post processing.

        Args:
            simulation (smuthi.simulation.Simulation):  simulation object containing input and solution of the problem
        """
        sys.stdout.write("Post processing ... ")
        sys.stdout.flush()
            
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
                
                if type(initial_field).__name__ == 'PlaneWave':
                    self.scattering_cross_section, self.extinction_cross_section = evaluate_cross_section(
                        initial_field=initial_field, particle_list=particle_list, layer_system=layer_system, 
                        outputdir=outputdir, show_plots=item.get('show plots', False), 
                        save_plots=item.get('save plots', False),
                        save_data=item.get('save data', False), length_unit=simulation.length_unit)
                elif type(initial_field).__name__ == 'GaussianBeam':
                    self.total_far_field, self.initial_far_field, self.scattered_far_field = sf.total_far_field(
                        initial_field=initial_field, particle_list=particle_list, layer_system=layer_system)    
                
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
                    if self.total_far_field.top() is not None:
                        top_pow = sum(self.total_far_field.top().integral()).real
                    else:
                        top_pow = 0
                    if self.total_far_field.bottom() is not None:
                        bottom_pow = sum(self.total_far_field.bottom().integral()).real
                    else:
                        bottom_pow = 0

                    print()
                    print('-------------------------------------------------------------------------')
                    print('Far field:')
                    print('Initial power:                                         ', in_pow)
                    if initial_field.polar_angle < np.pi / 2:
                        if bottom_pow:
                            print('Radiation into bottom layer (total reflection):        ', bottom_pow,
                                  ' or ', round(bottom_pow / in_pow * 100, 2), '%')
                        if top_pow:
                            print('Radiation into top layer (total transmission):         ', top_pow,
                                  ' or ', round(top_pow / in_pow * 100, 2), '%')
                    else:
                        if bottom_pow:
                            print('Radiation into bottom layer (total transmission):      ', bottom_pow,
                                  ' or ', round(bottom_pow / in_pow * 100, 2), '%')
                        if top_pow:
                            print('Radiation into top layer (total reflection):           ', top_pow,
                                  ' or ', round(top_pow / in_pow * 100, 2), '%')
                    print('Absorption and incoupling into waveguide modes:        ', in_pow - top_pow - bottom_pow,
                          ' or ', round((in_pow - top_pow - bottom_pow) / in_pow * 100, 2), '%')
                    print('-------------------------------------------------------------------------')
                elif (type(initial_field).__name__ == 'DipoleSource' 
                      or type(initial_field).__name__ == 'DipoleCollection'):
                    self.total_far_field, self.initial_far_field, self.scattered_far_field = sf.total_far_field(
                        initial_field=initial_field, particle_list=particle_list, layer_system=layer_system)    
                
                    go.show_far_field(far_field=self.total_far_field, save_plots=item.get('save plots', False),
                                      show_plots=item.get('show plots', False), save_data=item.get('save data', False),
                                      tag='total_far_field', outputdir=outputdir, flip_downward=True, split=True)
                    go.show_far_field(far_field=self.initial_far_field, save_plots=item.get('save plots', False),
                                      show_plots=item.get('show plots', False), save_data=item.get('save data', False),
                                      tag='initial_far_field', outputdir=outputdir, flip_downward=True, split=True)
                    go.show_far_field(far_field=self.scattered_far_field, save_plots=item.get('save plots', False),
                                      show_plots=item.get('show plots', False), save_data=item.get('save data', False),
                                      tag='scattered_far_field', outputdir=outputdir, flip_downward=True, split=True)
                    
                    if type(initial_field).__name__ == 'DipoleSource':
                        diss_pow = initial_field.dissipated_power(particle_list, layer_system)
                    else:
                        diss_pow = sum(initial_field.dissipated_power(particle_list, layer_system))
                    
                    assert abs(diss_pow.imag / diss_pow) < 1e-8 
                    diss_pow = diss_pow.real
                    
                    if self.total_far_field.top() is not None:
                        top_pow = sum(self.total_far_field.top().integral()).real
                    else:
                        top_pow = 0
                    if self.total_far_field.bottom() is not None:
                        bottom_pow = sum(self.total_far_field.bottom().integral()).real
                    else:
                        bottom_pow = 0

                    print()
                    print('-------------------------------------------------------------------------')
                    print('Dissipated power:                                  ', diss_pow)
                    print()
                    print('Far field:')
                    if bottom_pow:
                        print('Radiation into bottom layer (bottom outcoupling):  ', bottom_pow,
                              ' or ', round(bottom_pow / diss_pow * 100, 2), '%')
                    if top_pow:
                        print('Radiation into top layer (top outcoupling):        ', top_pow,
                              ' or ', round(top_pow / diss_pow * 100, 2), '%')
                    print('Absorption and incoupling into waveguide modes:    ', diss_pow - top_pow - bottom_pow,
                          ' or ', round((diss_pow - top_pow - bottom_pow) / diss_pow * 100, 2), '%')
                    print('-------------------------------------------------------------------------')
                    
            elif item['task'] == 'evaluate near field':
                sys.stdout.write("\nEvaluate near fields ... ")
                sys.stdout.flush()

                quantities_to_plot = item['quantities to plot']

                if simulation.output_dir:
                    outputdir = simulation.output_dir + '/near_field'
                else:
                    outputdir = '.'

                go.show_near_field(quantities_to_plot=quantities_to_plot, show_plots=item.get('show plots', False),
                                   save_plots=item.get('save plots', False), save_data=item.get('save data', False),
                                   save_animations=item.get('save animations', False), outputdir=outputdir,
                                   xmin=item.get('xmin', 0), xmax=item.get('xmax', 0), ymin=item.get('ymin', 0),
                                   ymax=item.get('ymax', 0), zmin=item.get('zmin', 0), zmax=item.get('zmax', 0),
                                   simulation=simulation, max_field=item.get('maximal field strength'),
                                   resolution_step=item.get('spatial resolution', 25),
                                   max_particle_distance=item.get('maximal particle distance', float('inf')),
                                   interpolate_step=item.get('interpolation spatial resolution'))

                sys.stdout.write("done. \n")
                sys.stdout.flush()


def evaluate_cross_section(polar_angles='default', azimuthal_angles='default', initial_field=None, particle_list=None, 
                           layer_system=None, outputdir=None, show_plots=None, save_plots=None, save_data=None, 
                           length_unit=None):
    """Compute differential scattering cross section as well as extinction cross sections.

    Args:
        polar_angles (numpy.ndarray or str):  array of polar angles for differential cross section.
                                              if 'default', use smuthi.coordinates.default_polar_angles
        azimuthal_angles (numpy.ndarray or str):    array of azimuthal angles for differential cross section
                                                    if 'default', use smuthi.coordinates.default_azimuthal_angles
        initial_field (smuthi.initial.InitialField):    initial field object
        particle_list (list):   list of smuthi.particles.Particle objects
        layer_system (smuthi.layers.LayerSystem): stratified medium
        outputdir (str):    path to folder where to store data and figures
        show_plots (bool):  show plots if true
        save_plots (bool):  export plots to disc if true
        save_data (bool):   save raw data to disc if true
        length_unit (str):  length unit used in simulation, e.g. 'nm'

    Returns:
        A tuple of a scattering cross section object and an extinction cross section dictionary.


    """
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
    
