# -*- coding: utf-8 -*-
"""Provide class for the representation of scattering particles."""
import smuthi.field_expansion as fldex
import smuthi.t_matrix as tmt
import smuthi.coordinates as coord
import smuthi.nfmds.t_matrix_axsym as nftaxs

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

        self.euler_angles=euler_angles
        self.refractive_index = refractive_index
        self.scattered_field.spherical_wave_expansion = fldex.SphericalWaveExpansion(l_max, m_max)
        self.scattered_field.spherical_wave_expansion.regular_coefficients = None
        self.incoming_field.spherical_wave_expansion = fldex.SphericalWaveExpansion(l_max, m_max)
        self.incoming_field.spherical_wave_expansion.outgoing_coefficients = None


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

        if position is None:
            self.position = [0, 0, 0]
        else:
            self.position = position

        Particle.__init__(self, position=position, refractive_index=refractive_index, l_max=l_max, m_max=m_max)
        self.radius = radius

    def compute_T_matrix(self, vacuum_wavelength, layer_system):
        iS = layer_system.layer_number(self.position[2])
        k_medium = coord.angular_frequency(vacuum_wavelength) * layer_system.refractive_indices[iS]
        k_particle = coord.angular_frequency(vacuum_wavelength) * self.refractive_index
        return tmt.t_matrix_sphere(k_medium, k_particle, self.radius, self.scattered_field.spherical_wave_expansion)


class Spheroid(Particle):
    """Particle subclass for spheroids.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        refractive_index (complex): Complex refractive index of particle
        semi_axis_c (float):        Spheroid half axis in direction of axis of revolution (z-axis if not rotated)
        semi_axis_a (float):        Spheroid half axis in lateral direction (x- and y-axis if not rotated)
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
        t_matrix_method (dict):     Dictionary containing the parameters for the algorithm to compute the T-matrix
    """
    def __init__(self, position=None, euler_angles=None, refractive_index=1+0j, semi_axis_c=1, semi_axis_a=1,
                 l_max=None, m_max=None, t_matrix_method=None):

        if position is None:
            self.position = [0, 0, 0]
        else:
            self.position = position

        if euler_angles is None:
            self.euler_angles = [0, 0, 0]
        else:
            self.euler_angles = euler_angles

        if t_matrix_method is None:
            self.t_matrix_method = {}
        else:
            self.t_matrix_method = t_matrix_method


        Particle.__init__(self, position=position, euler_angles=euler_angles, refractive_index=refractive_index,
                          l_max=l_max, m_max=m_max)
        self.semi_axis_c = semi_axis_c
        self.semi_axis_a = semi_axis_a


    def compute_T_matrix(self, vacuum_wavelength, layer_system):
        iS = layer_system.layer_number(self.position[2])
        lmax = self.scattered_field.spherical_wave_expansion.l_max
        t = nftaxs.tmatrix_spheroid(vacuum_wavelength=vacuum_wavelength,
                                    layer_refractive_index=layer_system.refractive_indices[iS],
                                    particle_refractive_index=self.refractive_index,
                                    semi_axis_c=self.semi_axis_c, semi_axis_a=self.semi_axis_a,
                                    use_ds=self.t_matrix_method.get('use discrete sources', True),
                                    nint=self.t_matrix_method.get('nint', 200),
                                    nrank=self.t_matrix_method.get('nrank', lmax + 2))
        return t

class FiniteCylinder(Particle):
    """Particle subclass for finite cylinders.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        refractive_index (complex): Complex refractive index of particle
        cylinder_radius (float):    Radius of cylinder (length unit)
        cylinder_height (float):    Height of cylinder, in z-direction if not rotated (length unit)
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
    """
    def __init__(self, position=None, euler_angles=None, refractive_index=1+0j, cylinder_radius=1,
                 cylinder_height=1, l_max=None, m_max=None, t_matrix_method=None):

        if position is None:
            self.position = [0, 0, 0]
        else:
            self.position = position

        if euler_angles is None:
            self.euler_angles = [0, 0, 0]
        else:
            self.euler_angles = euler_angles

        if t_matrix_method is None:
            self.t_matrix_method = {}
        else:
            self.t_matrix_method = t_matrix_method

        Particle.__init__(self, position=position, euler_angles=euler_angles, refractive_index=refractive_index,
                          l_max=l_max, m_max=m_max)
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        self.t_matrix_method = t_matrix_method

    def compute_T_matrix(self, vacuum_wavelength, layer_system):
        iS = layer_system.layer_number(self.position[2])
        method = self.t_matrix_method
        t = nftaxs.tmatrix_cylinder(vacuum_wavelength=vacuum_wavelength,
                                    layer_refractive_index=layer_system.refractive_indices[iS],
                                    particle_refractive_index=self.refractive_index,
                                    cylinder_height=self.cylinder_height, cylinder_radius=self.cylinder_radius,
                                    use_ds=method.get('use discrete sources', True), nint=method.get('nint', 200),
                                    nrank=method.get('nrank', self.scattered_field.spherical_wave_expansion.l_max + 2))
        return t

class ParticleCollection:
    """Collection of scattering particles."""
    def __init__(self, particle_list=None):
        """A list of dictionaries Particle objects"""
        if particle_list is None:
            self.particles = []
        else:
            self.particles = particle_list

    def add(self, particle):
        """Add particle to collection"""
        self.particles.append(particle)

    def remove_particle(self, i):
        """Remove i-th particle from collection"""
        del self.particles[i]

    def particle_number(self):
        """Return total number of particles in collection"""
        return len(self.particles)

    def particle_positions(self):
        """Return a list of particle positions"""
        return [self.particles[i].position for i in range(self.particle_number())]

    def compute_T_matrices(self, layer_system, vacuum_wavelength, method):