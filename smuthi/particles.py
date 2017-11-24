# -*- coding: utf-8 -*-
"""Provide class for the representation of scattering particles."""
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

