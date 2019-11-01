# -*- coding: utf-8 -*-
"""Provide class for the representation of scattering particles."""

import smuthi.fields.expansions as fldex
import smuthi.linearsystem.tmatrix as tmt
import numpy as np


class Particle:
    """Base class for scattering particles.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        euler_angles (list):        Particle Euler angles in the format [alpha, beta, gamma]
        refractive_index (complex): Complex refractive index of particle
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
    """
    def __init__(self, position=None, euler_angles=None, refractive_index=1+0j, l_max=None, m_max=None):
        
        if position is None:
            self.position = [0, 0, 0]
        else:
            self.position = position

        if euler_angles is None:
            self.euler_angles = [0, 0, 0]
        else:
            self.euler_angles = euler_angles

        self.refractive_index = refractive_index
        self.l_max = l_max
        if m_max is not None:
            self.m_max = m_max
        else:
            self.m_max = l_max
        self.initial_field = None
        self.scattered_field = None
        self.t_matrix = None
        
    def circumscribing_sphere_radius(self):
        """Virtual method to be overwritten"""
        pass
    
    def automated_lmax_mmax_selection(self, vacuum_wavelength, ambient_medium,
                                      lmax_stop=20, max_rel_diff=1e-3):
        """ Automated selection of a particle's maximal multipole degree lmax and maximal multipole order mmax. 
        
        Args:
            vacuum_wavelength (float):          Vacuum wavelength :math:`\lambda` (length unit)
            ambient_medium (complex):           Complex refractive index of the particle's ambient medium.
            lmax_stop (int):                    Maximal multipole degree to be considered.
            max_rel_diff (flaot):               Maximal relative difference between T-matrices of successive lmax and mmax
                                                that is tolerated (decission criterion).
        """
        def relative_difference_Tmatrices(Tmat_s, lmax_s, mmax_s, Tmat_l, lmax_l, mmax_l):
            n_list_s = np.zeros(len(Tmat_s), dtype=int)
            n_list_l = np.zeros(len(Tmat_s), dtype=int)
            idx = 0
            for tau in range(2):
                for l in range(1, lmax_s + 1):
                    for m in range(np.max([-l, -mmax_s]), np.min([l, mmax_s]) + 1):        
                        n_list_s[idx] = fldex.multi_to_single_index(tau, l, m, lmax_s, mmax_s) 
                        n_list_l[idx] = fldex.multi_to_single_index(tau, l, m, lmax_l, mmax_l)                     
                        idx += 1                                
            row, column = np.meshgrid(n_list_s, n_list_s)
            row2, column2 = np.meshgrid(n_list_l, n_list_l)
            TMat_temp = np.zeros([len(Tmat_l), len(Tmat_l)], dtype=complex)
            TMat_temp[row2, column2] = Tmat_s[row, column]
            return np.linalg.norm(Tmat_l - TMat_temp) / np.linalg.norm(Tmat_l)
           
        L2_norm = [[], []]
        TMatrix = [[], []]
        self.l_max = 0
        lmax_decision, mmax_decision = False, False
        # increase lmax until either Csca or each element of the DSCS does not change significantly  
        while not lmax_decision:
            self.l_max += 1
            self.m_max = self.l_max # for finding the lmax keep lmax = mmax           
            TMatrix[0].append(tmt.t_matrix(vacuum_wavelength, ambient_medium, self))     
            if len(TMatrix[0]) > 1: # do we have at least two values to compare     
                L2_norm[0].append(relative_difference_Tmatrices(TMatrix[0][-2], self.l_max - 1, self.m_max - 1,
                                                                TMatrix[0][-1], self.l_max, self.m_max))
                if L2_norm[0][-1] <= max_rel_diff: # condition satisfied?
                    lmax_decision = True
                    self.l_max -= 1
            assert self.l_max <= lmax_stop, 'The set precision requires lmax > %d.' % lmax_stop
        
        self.m_max = 0    # now find the corresponding mmax
        while not mmax_decision:
            self.m_max += 1
            if self.m_max == self.l_max:   # mmax = lmax is valid in any case
                mmax_decision = True
                self.t_matrix = TMatrix[0][-2]
            else:
                TMatrix[1].append(tmt.t_matrix(vacuum_wavelength, ambient_medium, self))           
                L2_norm[1].append(relative_difference_Tmatrices(TMatrix[1][-1], self.l_max, self.m_max,
                                                                TMatrix[0][-2], self.l_max, self.l_max))
                if L2_norm[1][-1] <= max_rel_diff:
                    mmax_decision = True
                    self.t_matrix = TMatrix[1][-1]
        print(type(self), '\n',
              'lmax has been set to %d ' % self.l_max, '\n',
              'mmax has been set to %d ' % self.m_max)


class Sphere(Particle):
    """Particle subclass for spheres.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        refractive_index (complex): Complex refractive index of particle
        radius (float):             Particle radius (length unit)
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
        t_matrix_method (dict):     Dictionary containing the parameters for the algorithm to compute the T-matrix
    """
    def __init__(self, position=None, refractive_index=1+0j, radius=1, l_max=None, m_max=None):

        Particle.__init__(self, position=position, refractive_index=refractive_index, l_max=l_max, m_max=m_max)

        self.radius = radius
        
    def circumscribing_sphere_radius(self):
        return self.radius


class Spheroid(Particle):
    """Particle subclass for spheroids.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        euler_angles (list):        Euler angles [alpha, beta, gamma] in (zy'z''-convention) in radian.
                                    Alternatively, you can specify the polar and azimuthal angle of the axis of 
                                    revolution.
        polar_angle (float):        Polar angle of axis of revolution. 
        azimuthal_angle (float):    Azimuthal angle of axis of revolution.
        refractive_index (complex): Complex refractive index of particle
        semi_axis_c (float):        Spheroid half axis in direction of axis of revolution (z-axis if not rotated)
        semi_axis_a (float):        Spheroid half axis in lateral direction (x- and y-axis if not rotated)
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
        t_matrix_method (dict):     Dictionary containing the parameters for the algorithm to compute the T-matrix
    """
    def __init__(self, position=None, euler_angles=None, polar_angle=0, azimuthal_angle=0, refractive_index=1+0j, 
                 semi_axis_c=1, semi_axis_a=1, l_max=None, m_max=None, t_matrix_method=None):

        if euler_angles is None:
            euler_angles = [azimuthal_angle, polar_angle, 0]
            
        Particle.__init__(self, position=position, euler_angles=euler_angles, refractive_index=refractive_index,
                          l_max=l_max, m_max=m_max)
        
        if t_matrix_method is None:
            self.t_matrix_method = {}
        else:
            self.t_matrix_method = t_matrix_method

        self.semi_axis_c = semi_axis_c
        self.semi_axis_a = semi_axis_a

    def circumscribing_sphere_radius(self):
        return max([self.semi_axis_a, self.semi_axis_c])


class FiniteCylinder(Particle):
    """Particle subclass for finite cylinders.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        euler_angles (list):        Euler angles [alpha, beta, gamma] in (zy'z''-convention) in radian.
                                    Alternatively, you can specify the polar and azimuthal angle of the axis of 
                                    revolution.
        polar_angle (float):        Polar angle of axis of revolution. 
        azimuthal_angle (float):    Azimuthal angle of axis of revolution.
        refractive_index (complex): Complex refractive index of particle
        cylinder_radius (float):    Radius of cylinder (length unit)
        cylinder_height (float):    Height of cylinder, in z-direction if not rotated (length unit)
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
    """
    def __init__(self, position=None, euler_angles=None, polar_angle=0, azimuthal_angle=0, refractive_index=1+0j, 
                 cylinder_radius=1, cylinder_height=1, l_max=None, m_max=None, t_matrix_method=None):

        if euler_angles is None:
            euler_angles = [azimuthal_angle, polar_angle, 0]

        Particle.__init__(self, position=position, euler_angles=euler_angles, refractive_index=refractive_index,
                          l_max=l_max, m_max=m_max)

        if t_matrix_method is None:
            self.t_matrix_method = {}
        else:
            self.t_matrix_method = t_matrix_method

        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        
    def circumscribing_sphere_radius(self):
        return np.sqrt((self.cylinder_height / 2)**2 + self.cylinder_radius**2)
