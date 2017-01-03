# -*- coding: utf-8 -*-
"""Provide class for the representation of scattering particles."""


class ParticleCollection:
    """Collection of scattering particles."""
    # Particle templates, in the format of a list of dictionaries, each element corresponds to one
    # particle type:
    specs_list = []

    # Maps particle number i to the index of its specs in the specs_list:
    specs_indices = []

    # A list of positions in the format [x,y,z]
    positions = []

    # A list of euler angles in the format [alpha,beta,gamma]
    euler_angles = []

    def add_sphere(self, radius, refractive_index, x, y, z):
        """Add a sphere to the collection.

        input parameters:
        radius:             sphere radius (length unit)
        refractive_index:   complex refractive index in the form n+jk
        x:                  center x position (length unit)
        y:                  center y position (length unit)
        z:                  center z position, normal to layer interfaces (length unit)
        """
        for i, specs in enumerate(self.specs_list):
            # Check if particle specs are already in specs list:
            if (specs['shape'] == 'sphere' and specs['radius'] == radius and
                        specs['refractive index'] == refractive_index):
                self.specs_indices.append(i)
                self.positions.append([x, y, z])
                self.euler_angles.append([0, 0, 0])
                break
        else:  # Add entry in specs list
            self.specs_list.append({'shape': 'sphere', 'radius': radius, 'refractive index': refractive_index})
            self.specs_indices.append(len(self.specs_list) - 1)
            self.positions.append([x, y, z])
            self.euler_angles.append([0, 0, 0])

    def add_spheroid(self, semi_axis_c, semi_axis_a, refractive_index, x, y, z, alpha=0, beta=0, gamma=0):
        """Add a spheroid to the collection.

        input parameters:
        semi_axis_c:        spheroid semi axis along symmetry axis (length unit)
        semi_axis_a:        spheroid semi axis a=b in transverse direction (length unit)
        refractive_index:   complex refractive index in the form n+jk
        x:                  center x position (length unit)
        y:                  center y position (length unit)
        z:                  center z position, normal to layer interfaces (length unit)
        alpha:              todo, default=0
        beta:               todo, default=0
        beta:               todo, default=0
        """
        for i, specs in enumerate(self.specs_list):
            if (specs['shape'] == 'spheroid' and specs['semi axis c'] == semi_axis_c and
                        specs['semi axis a'] == semi_axis_a and specs['refractive index'] == refractive_index):
                self.specs_indices.append(i)
                self.positions.append([x, y, z])
                self.euler_angles.append([alpha, beta, gamma])
                break
        else:
            self.specs_list.append({'shape': 'spheroid', 'semi axis c': semi_axis_c, 'semi axis a': semi_axis_a,
                                    'refractive index': refractive_index})
            self.specs_indices.append(len(self.specs_list) - 1)
            self.positions.append([x, y, z])
            self.euler_angles.append([alpha, beta, gamma])

    def add_finite_cylinder(self, cylinder_radius, cylinder_height, refractive_index, x, y, z, alpha=0, beta=0,
                            gamma=0):
        """Add finite cylinder to the collection.

        input parameters:
        cylinder_radius:    cylinder radius (length unit)
        cylinder_height:    height of cylinder (length unit)
        refractive_index:   complex refractive index in the form n+jk
        x:                  center x position (length unit)
        y:                  center y position (length unit)
        z:                  center z position, normal to layer interfaces (length unit)
        alpha:              todo, default=0
        beta:               todo, default=0
        beta:               todo, default=0
        """
        for i, specs in enumerate(self.specs_list):
            if (specs['shape'] == 'finite cylinder' and specs['cylinder radius'] == cylinder_radius and
                        specs['cylinder height'] == cylinder_height and specs['refractive index'] == refractive_index):
                self.specs_indices.append(i)
                self.positions.append([x, y, z])
                self.euler_angles.append([alpha, beta, gamma])
                break
        else:
            self.specs_list.append(
                {'shape': 'finite cylinder', 'cylinder radius': cylinder_radius, 'cylinder height': cylinder_height,
                 'refractive index': refractive_index})
            self.specs_indices.append(len(self.specs_list) - 1)
            self.positions.append([x, y, z])
            self.euler_angles.append([alpha, beta, gamma])

    def remove_particle(self, i):
        """Remove i-th particle from collection"""
        del(self.specs_indices[i])
        del(self.positions[i])
        del(self.euler_angles[i])

    def particle_specs(self, i):
        """Return particle specs of particle i"""
        return self.specs_list[self.specs_indices[i]]

    def particle_number(self):
        """Return total number of particles in collection"""
        return len(self.specs_indices)
