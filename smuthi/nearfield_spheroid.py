# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:38:51 2018

@author: theobald2
"""
import numpy as np
import scipy
import smuthi.coordinates as coord
import smuthi.field_expansion as fldex
import matplotlib.pyplot as plt
import smuthi.graphical_output as graph
import smuthi.layers as lay


def spheroid_highest_lowest_surface_points(ab_halfaxis, c_halfaxis, center, orientation):
    """
    Computation of a spheroids surface points with the highest / lowest z-value
    
    Args:
        ab_halfaxis1 (float):        Half axis orthogonal to symmetry axis of the spheroid 
        c_halfaxis1 (float):         Half axis parallel to symmetry axis of the spheroid 
        center (numpy.array):        Center coordinates of the spheroid 
        orientation (numpy.array):   Orientation angles of the spheroid
        
    Retruns:
        Tuple containing:
          - surface point with the highest z-value (numpy.array)
          - surface point with the lowest z-value (numpy.array)
    """
        
    def rotation_matrix(ang):
        rot_mat = (np.array([[np.cos(ang[0]) * np.cos(ang[1]), -np.sin(ang[0]), np.cos(ang[0]) * np.sin(ang[1])],
                        [np.sin(ang[0]) * np.cos(ang[1]), np.cos(ang[0]), np.sin(ang[0]) * np.sin(ang[1])],
                        [-np.sin(ang[1]), 0, np.cos(ang[1])]]))
        return rot_mat
    
    rot_matrix = rotation_matrix(orientation)
    eigenvalue_matrix = np.array([[1 / ab_halfaxis ** 2, 0, 0], [0, 1 / ab_halfaxis ** 2, 0], [0, 0, 1 / c_halfaxis ** 2]])
    E = np.dot(rot_matrix, np.dot(eigenvalue_matrix, np.transpose(rot_matrix)))
    L = np.linalg.cholesky(E)
    L_inv_trans = np.linalg.inv(np.transpose(L))
    
    lambd = 0.5 * ((L_inv_trans[2,0]**2 + L_inv_trans[2,1]**2 + L_inv_trans[2,2]**2) ** 0.5)
    
    y = np.array(np.zeros([3,1], dtype=float))
    y[0], y[1], y[2] = L_inv_trans[2,0] / (2 * lambd), L_inv_trans[2,1] / (2 * lambd), L_inv_trans[2,2] / (2 * lambd)
    
    r_highz = np.array(np.zeros([1,3], dtype=float))
    r_lowz = np.array(np.zeros([1,3], dtype=float))
    r_highz[0,0] = L_inv_trans[0,0] * y[0] + L_inv_trans[0,1] * y[1] + L_inv_trans[0,2] * y[2] + center[0]
    r_highz[0,1] = L_inv_trans[1,0] * y[0] + L_inv_trans[1,1] * y[1] + L_inv_trans[1,2] * y[2] + center[1]
    r_highz[0,2] = L_inv_trans[2,0] * y[0] + L_inv_trans[2,1] * y[1] + L_inv_trans[2,2] * y[2] + center[2]
    
  
    r_lowz[0,0] = L_inv_trans[0,0] * -y[0] + L_inv_trans[0,1] * -y[1] + L_inv_trans[0,2] * -y[2] + center[0]
    r_lowz[0,1] = L_inv_trans[1,0] * -y[0] + L_inv_trans[1,1] * -y[1] + L_inv_trans[1,2] * -y[2] + center[1]
    r_lowz[0,2] = L_inv_trans[2,0] * -y[0] + L_inv_trans[2,1] * -y[1] + L_inv_trans[2,2] * -y[2] + center[2]
    
    return r_highz, r_lowz


def spheroid_closest_surface_point(ab_halfaxis, c_halfaxis, center, orientation, coordinate):
    """
    Computation of a spheroids surface point, that is closest to a given reference coordinate
    
    Args:
        ab_halfaxis1 (float):        Half axis orthogonal to symmetry axis of the spheroid 
        c_halfaxis1 (float):         Half axis parallel to symmetry axis of the spheroid 
        center (numpy.array):        Center coordinates of the spheroid 
        orientation (numpy.array):   Orientation angles of the spheroid
        coordinate (numpy.array):    Reference point 
        
    Retruns:
        Tuple containing:
          - surface point closest to the reference coordinate (numpy.array)
          - first rotation Euler angle alpha (float)
          - second rotation Euler angle beta (float)
    """

# This function has a problem with coordinates, that are inside the spheroid (results in an infinite loop of wrong minima).
        
    def rotation_matrix(ang):
        rot_mat = (np.array([[np.cos(ang[0]) * np.cos(ang[1]), -np.sin(ang[0]), np.cos(ang[0]) * np.sin(ang[1])],
                  [np.sin(ang[0]) * np.cos(ang[1]), np.cos(ang[0]), np.sin(ang[0]) * np.sin(ang[1])],
                  [-np.sin(ang[1]), 0, np.cos(ang[1])]]))
        return rot_mat
    
    rot_matrix = rotation_matrix(orientation)
    eigenvalue_matrix = np.array([[1 / ab_halfaxis ** 2, 0, 0], [0, 1 / ab_halfaxis ** 2, 0], [0, 0, 1 / c_halfaxis ** 2]])
    E = np.dot(rot_matrix, np.dot(eigenvalue_matrix, np.transpose(rot_matrix)))
    L = np.linalg.cholesky(E)
    
    H = np.dot(np.linalg.inv(L), np.transpose(np.linalg.inv(L)))
    f = np.dot(np.transpose(center - coordinate), np.transpose(np.linalg.inv(L)))
        
    def minimization_fun(y_vec):
        fun = 0.5 * np.dot(np.dot(np.transpose(y_vec), H), y_vec) + np.dot(f, y_vec)
        return fun
    def constraint_fun(x):
        eq_constraint = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5 - 1
        return eq_constraint
    bnds = ((-1, 1), (-1, 1), (-1, 1))
    length_constraints = {'type' : 'eq', 'fun' : constraint_fun}
       
    flag = False
    while flag == False:
        x0 = -1 + np.dot((1 + 1), np.random.rand(3))
        optimization_result = scipy.optimize.minimize(minimization_fun, x0, method='SLSQP', bounds=bnds,
                                                      constraints=length_constraints, tol=None, callback=None, options=None)
        p1 = np.transpose(np.dot(np.transpose(np.linalg.inv(L)), optimization_result['x']) + np.transpose(center))
        if optimization_result['success'] == True:
            if np.linalg.norm(p1 - coordinate) < np.linalg.norm(center - coordinate):
                flag = True
            else:
                print('wrong minimum ...')
        else:
            print('No minimum found ...')
    
    p1p2 = coordinate - p1
    azimuth = np.arctan2(p1p2[1], p1p2[0])
    elevation = np.arctan2(p1p2[2], (p1p2[0] ** 2 + p1p2[1] ** 2) ** 0.5)

    if p1p2[2] < 0:
        beta = (np.pi / 2) + elevation
    else:
        beta = (-np.pi / 2) + elevation
    alpha = -azimuth
    
    return p1, alpha, beta
    
 
    
def nearfield_pwe(particle, coordinate, k_parallel, azimuthal_angles, layer_system=None, layer_number=None):

# is the coordiate located outside the spheroid's circumscribing sphere?    
    if (np.linalg.norm(np.abs(coordinate - particle.position)) / np.max([particle.semi_axis_a, particle.semi_axis_c])) >= 1:
        ex, ey, ez = (particle.scattered_field.electric_field(coordinate[0], coordinate[1], coordinate[2])[0],
                      particle.scattered_field.electric_field(coordinate[0], coordinate[1], coordinate[2])[1], 
                      particle.scattered_field.electric_field(coordinate[0], coordinate[1], coordinate[2])[2])
        E = [ex, ey, ez]
    else:
        r_highz, r_lowz = spheroid_highest_lowest_surface_points(particle.semi_axis_a, particle.semi_axis_c, particle.position, particle.euler_angles)

# is the coordinate's z-value located between the highest and lowest point of the spheroid?    
        if coordinate[2] > r_lowz[0,2] and coordinate[2] < r_highz[0,2]:
            p1, alpha, beta = spheroid_closest_surface_point(particle.semi_axis_a, particle.semi_axis_c, particle.position, particle.euler_angles, coordinate)


            coordref = coordinate - particle.scattered_field.reference_point 
            coordref_prime = coord.vector_rotation(coordref, euler_angles=[-alpha, -beta, 0])
            coordinate_prime = particle.scattered_field.reference_point + coordref_prime
#            p1ref = p1 - particle.scattered_field.reference_point
#            p1ref_prime = coord.vector_rotation(p1ref, euler_angles=[-alpha, -beta, 0])
#            p1_prime = particle.scattered_field.reference_point + p1ref_prime
            
            b_prime = np.dot(np.transpose(fldex.block_rotation_matrix_D_svwf(particle.l_max, particle.m_max, -alpha, -beta, 0)), particle.scattered_field.coefficients)
            swe_prime = fldex.SphericalWaveExpansion(k=particle.scattered_field.k, l_max=particle.scattered_field.l_max,
                                                     m_max=particle.scattered_field.m_max, kind='outgoing', 
                                                     reference_point=particle.scattered_field.reference_point, lower_z=-np.inf, upper_z=np.inf, 
                                                     inner_r=0, outer_r=np.inf)
            swe_prime.coefficients = b_prime
# since the pwe_primes receives the same layer system as the original one, it has trouble with points that are very far away (in x and y direction)
# yet, the problem does not occure since, the swe is used when a point is outside the circumscribing sphere            
            pwe_prime = fldex.swe_to_pwe_conversion(swe=swe_prime, k_parallel='default', azimuthal_angles='default', layer_system=layer_system,
                                        layer_number=layer_number, layer_system_mediated=False)

            if coordinate_prime[2] >= particle.scattered_field.reference_point[2]:
                ex_prime, ey_prime, ez_prime = (pwe_prime[0].electric_field(coordinate_prime[0], coordinate_prime[1], coordinate_prime[2])[0],
                                                pwe_prime[0].electric_field(coordinate_prime[0], coordinate_prime[1], coordinate_prime[2])[1], 
                                                pwe_prime[0].electric_field(coordinate_prime[0], coordinate_prime[1], coordinate_prime[2])[2])
            else:
                ex_prime, ey_prime, ez_prime = (pwe_prime[1].electric_field(coordinate_prime[0], coordinate_prime[1], coordinate_prime[2])[0],
                                                pwe_prime[1].electric_field(coordinate_prime[0], coordinate_prime[1], coordinate_prime[2])[1], 
                                                pwe_prime[1].electric_field(coordinate_prime[0], coordinate_prime[1], coordinate_prime[2])[2])                                   

            E_prime = np.array([ex_prime, ey_prime, ez_prime])
            E = coord.inverse_vector_rotation(E_prime, euler_angles=[-alpha, -beta, 0])
            
        else:
            pwe = fldex.swe_to_pwe_conversion(swe=particle.scattered_field, k_parallel='default', azimuthal_angles='default', layer_system=layer_system,
                                              layer_number=layer_number, layer_system_mediated=False)
            if coordinate[2] >= particle.position[2]:
                ex, ey, ez = (pwe[0].electric_field(coordinate[0], coordinate[1], coordinate[2])[0],
                              pwe[0].electric_field(coordinate[0], coordinate[1], coordinate[2])[1], 
                              pwe[0].electric_field(coordinate[0], coordinate[1], coordinate[2])[2])
            else:
                ex, ey, ez = (pwe[1].electric_field(coordinate[0], coordinate[1], coordinate[2])[0],
                              pwe[1].electric_field(coordinate[0], coordinate[1], coordinate[2])[1], 
                              pwe[1].electric_field(coordinate[0], coordinate[1], coordinate[2])[2])   
                E = [ex, ey, ez]        
    erg = [E] 
             
    return erg
    
    
def fieldpoints(xmin, xmax, ymin, ymax, zmin, zmax, resolution):
    """
    Creates an 4-dimensional array to handle the nearfield computation via plane waves
    Args:
        xmin (float):       Plot from that x (length unit)
        xmax (float):       Plot up to that x (length unit)
        ymin (float):       Plot from that y (length unit)
        ymax (float):       Plot up to that y (length unit)
        zmin (float):       Plot from that z (length unit)
        zmax (float):       Plot up to that z (length unit)
        resolution (float):     Compute the field with that spatial resolution (length unit)
    Returns:
        fp (numpy.array):       4-dimensional array 
                                dimension 1,2,3 contains the x,y,z coordinate of all field points
                                dimension 4 contains a 'bool' whether E has been computed or not
    """      
    if xmin == xmax:
        dim1vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        yarr, zarr = np.meshgrid(dim1vec, dim2vec)
        xarr = yarr - yarr + xmin
    elif ymin == ymax:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        xarr, zarr = np.meshgrid(dim1vec, dim2vec)
        yarr = xarr - xarr + ymin
    else:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        xarr, yarr = np.meshgrid(dim1vec, dim2vec)
        zarr = xarr - xarr + zmin

    fp = np.empty((dim1vec.size, dim2vec.size, 4), dtype=object)   
    fp[:, :, 0], fp[:, :, 1], fp[:, :, 2] = xarr, yarr, zarr
    fp[:, :, 3] = np.zeros((dim1vec.size, dim2vec.size), dtype=bool)
    
    return fp, dim1vec, dim2vec 
    
def inside_particles(fieldpoints, particle_list):
    """
    Checks for each field point, whether it is located inside one of the particles.
    Args:
        fieldpoints(numpy.array):       4-dimensional numpy.array that contains the x,y,z-coordinate of each fieldpoint 
                                        and a bool, to determin wether this fieldpoint needs to be computed
        particle_list:                  Object of smuthi.particle_list that contains all information about the scattering particles.
    Returns:
        fieldpoints(numpy.array):       The 4th dimension now contains the information whether the coordinate is inside a particle.
                                        If so, the scattered field does not need to be computed.
    """
        
    def rotation_matrix(ang):
        rot_mat = (np.array([[np.cos(ang[0]) * np.cos(ang[1]), -np.sin(ang[0]), np.cos(ang[0]) * np.sin(ang[1])],
                  [np.sin(ang[0]) * np.cos(ang[1]), np.cos(ang[0]), np.sin(ang[0]) * np.sin(ang[1])],
                  [-np.sin(ang[1]), 0, np.cos(ang[1])]]))
        return rot_mat

    for p in range(np.size(particle_list)):    
        rot_matrix = rotation_matrix(particle_list[p].euler_angles)
        eigenvalue_matrix = np.array([[1 / particle_list[p].semi_axis_a ** 2, 0, 0],
                                      [0, 1 / particle_list[p].semi_axis_a ** 2, 0],
                                      [0, 0, 1 / particle_list[p].semi_axis_c ** 2]])
        E = np.dot(rot_matrix, np.dot(eigenvalue_matrix, np.transpose(rot_matrix)))
        L = np.linalg.cholesky(E)
        S = np.matrix.getH(L)
        for k in range(np.size(fieldpoints[:, 0, 0])):
            for l in range(np.size(fieldpoints[0, :, 0])):
                if not fieldpoints[k, l, 3]:
                    coord_prime = -(np.dot(S, (particle_list[p].position - np.array([fieldpoints[k, l, 0], fieldpoints[k, l, 1], fieldpoints[k, l, 2]]))))
                    if np.round(np.linalg.norm(coord_prime), 5) <= 1:
                        fieldpoints[k, l, 3] = 'True'

    return fieldpoints    
    
def pwe_nearfield_superposition(xmin, xmax, ymin, ymax, zmin, zmax, resolution, k_parallel='default', azimuthal_angles='default',
                       simulation=None):
    
    fp0, dim1vec, dim2vec = fieldpoints(xmin, xmax, ymin, ymax, zmin, zmax, resolution)
    field_list = np.empty((np.size(simulation.particle_list), dim1vec.size, dim2vec.size, 4), dtype=object)
    Ex = np.zeros([np.size(dim1vec), np.size(dim2vec)], dtype=complex)
    Ey, Ez = Ex, Ex
    fp = inside_particles(fp0, simulation.particle_list)
    for i in range(np.size(simulation.particle_list)):
        field_list[i, :, :, 0] = np.zeros((dim1vec.size, dim2vec.size), dtype=complex)
        field_list[i, :, :, 1], field_list[i, :, :, 2] = field_list[i, :, :, 0], field_list[i, :, :, 0]        
        field_list[i, :, :, 3] = fp[:, :, 3]
# all points outside the circumscribing sphere
        temp_array = np.array([0, 0, 0, 0, 0], dtype=float)[None,:]
        for k in range(np.size(fp[:, 0, 0])):
            for l in range(np.size(fp[0, :, 0])):
                if not field_list[i, k, l, 3]:
                    if (np.linalg.norm(np.abs(np.array([fp[k, l, 0] - simulation.particle_list[i].position[0], 
                                                        fp[k, l, 1] - simulation.particle_list[i].position[1],
                                                        fp[k, l, 2] - simulation.particle_list[i].position[2]])))
                        / np.max([simulation.particle_list[i].semi_axis_a, simulation.particle_list[i].semi_axis_c])) >= 1: # this should be a 1
                       temp_array = np.append(temp_array, np.array([k , l, fp[k, l, 0], fp[k, l, 1], fp[k, l, 2]])[None,:],
                                              axis=0)

        temp_array = np.delete(temp_array, 0, 0)
        ex, ey, ez = (simulation.particle_list[i].scattered_field.electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0],
                      simulation.particle_list[i].scattered_field.electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1], 
                      simulation.particle_list[i].scattered_field.electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2])
        for p in range(np.size(temp_array, axis=0)):
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 0] = ex[p]
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 1] = ey[p]
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 2] = ez[p]
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 3] = 'True'
        if ex.any():
            del ex, ey, ez               
# all points inside the circumscribing sphere and above/below the highest/lowest point of the spheroid  
#        waypoints = [0, 0.8, 0.8-0.1j, 2.1-0.1j, 2.1, 3]
#        neff_discr = 2e-2
#        wl = 2*np.pi/simulation.particle_list[i].initial_field.k
#        coord.set_default_k_parallel(vacuum_wavelength = wl, neff_waypoints=waypoints, neff_resolution=neff_discr)
        r_highz, r_lowz = spheroid_highest_lowest_surface_points(simulation.particle_list[i].semi_axis_a, simulation.particle_list[i].semi_axis_c,
                                                                 simulation.particle_list[i].position, simulation.particle_list[i].euler_angles)
        temp_array = np.array([0, 0, 0, 0, 0], dtype=float)[None,:]
        for k in range(np.size(fp[:, 0, 0])):
            for l in range(np.size(fp[0, :, 0])):
                if not field_list[i, k, l, 3]:
                    if fp[k, l, 2] <= r_lowz[0, 2] or fp[k, l, 2] >= r_highz[0, 2]:
                        temp_array = np.append(temp_array, np.array([k, l, fp[k, l, 0], fp[k, l, 1], fp[k, l, 2]])[None,:], axis=0)
        
        temp_array = np.delete(temp_array, 0, 0)
        pwe = fldex.swe_to_pwe_conversion(swe=simulation.particle_list[i].scattered_field, k_parallel='default', azimuthal_angles='default',
                                          layer_system=simulation.layer_system,
                                          layer_number=simulation.layer_system.layer_number(simulation.particle_list[i].position[2]),
                                          layer_system_mediated=False)

        ex, ey, ez = (pwe[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0]
                      + pwe[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0],
                      pwe[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1]
                      + pwe[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1], 
                      pwe[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2]
                      + pwe[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2])
        
        for p in range(np.size(temp_array, axis=0)):
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 0] = ex[p]
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 1] = ey[p]
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 2] = ez[p]
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 3] = 'True'
        if ex.any():
            del ex, ey, ez   
        
# all points that are inside the circumscribing sphere and that need a rotation        
        while not np.array(field_list[i, :, :, 3], dtype=bool).all():
            for k in range(np.size(fp[:, 0, 0])):
                for l in range(np.size(fp[0, :, 0])):
                    if not field_list[i, k, l, 3]:
                        coordinate = np.array([fp[k, l, 0], fp[k, l, 1], fp[k, l, 2]])                  
                        p1, alpha, beta = spheroid_closest_surface_point(simulation.particle_list[i].semi_axis_a,
                                                                               simulation.particle_list[i].semi_axis_c,
                                                                               simulation.particle_list[i].position,
                                                                               simulation.particle_list[i].euler_angles, coordinate)
                        r_highz, r_lowz = spheroid_highest_lowest_surface_points(simulation.particle_list[i].semi_axis_a,
                                                                                 simulation.particle_list[i].semi_axis_c,
                                                                                 simulation.particle_list[i].position,
                                                                                 simulation.particle_list[i].euler_angles + np.array([-alpha, -beta, 0]))
                        b_prime = np.dot(np.transpose(fldex.block_rotation_matrix_D_svwf(simulation.particle_list[i].l_max,
                                         simulation.particle_list[i].m_max, -alpha, -beta, 0)), simulation.particle_list[i].scattered_field.coefficients)
                        swe_prime = fldex.SphericalWaveExpansion(k=simulation.particle_list[i].scattered_field.k,
                                                                 l_max=simulation.particle_list[i].scattered_field.l_max,
                                                                 m_max=simulation.particle_list[i].scattered_field.m_max, kind='outgoing', 
                                                                 reference_point=simulation.particle_list[i].position, lower_z=-np.inf, upper_z=np.inf, 
                                                                 inner_r=0, outer_r=np.inf)
                        swe_prime.coefficients = b_prime
                        pwe_prime = fldex.swe_to_pwe_conversion(swe=swe_prime, k_parallel='default', azimuthal_angles='default',
                                                                layer_system=simulation.layer_system,
                                                                layer_number=simulation.layer_system.layer_number(simulation.particle_list[i].position[2]),
                                                                layer_system_mediated=False)
                        
                        temp_array = np.array([0, 0, 0, 0, 0], dtype=float)[None,:]
                        for m in range(np.size(fp[:, 0, 0])):
                            for n in range(np.size(fp[0, :, 0])):
                                if not field_list[i, m, n, 3]:
                                    coordref = (np.array([fp[m, n, 0], fp[m, n, 1], fp[m, n, 2]])
                                                - np.array(simulation.particle_list[i].position))
                                    coordref_prime = coord.vector_rotation(coordref, euler_angles=[-alpha, -beta, 0])
                                    coord_prime = coordref_prime + simulation.particle_list[i].position
                                    if coord_prime[2] <= r_lowz[0, 2] or coord_prime[2] >= r_highz[0, 2]:
                                        temp_array = np.append(temp_array, np.array([m, n, coord_prime[0], coord_prime[1], coord_prime[2]])[None,:],
                                                               axis=0)
                        temp_array = np.delete(temp_array, 0, 0)                            
                        ex, ey, ez = (pwe_prime[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0]
                                      + pwe_prime[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0],
                                      pwe_prime[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1]
                                      + pwe_prime[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1], 
                                      pwe_prime[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2]
                                      + pwe_prime[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2])
                        for p in range(np.size(temp_array, axis=0)):
                            E_prime = coord.inverse_vector_rotation(np.array([ex[p], ey[p], ez[p]]), euler_angles=[-alpha, -beta, 0])
                            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 0] = E_prime[0]    
                            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 1] = E_prime[1]
                            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 2] = E_prime[2]
                            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 3] = 'True'
                        if ex.any():
                            del ex, ey, ez, E_prime
                   
        Ex = Ex + np.array(field_list[i, :, :, 0], dtype=complex)
        Ey = Ey + np.array(field_list[i, :, :, 1], dtype=complex)
        Ez = Ez + np.array(field_list[i, :, :, 2], dtype=complex)
    
    
    
    return field_list, Ey, dim1vec, dim2vec


#def cut_ellipse(semi_axis_a, semi_axis_c, center, orientation, plane):    
##   http://forum.diegeodaeten.de/index.php?mode=thread&id=4578 
#    def rotation_matrix(ang):
#        rot_mat = (np.array([[np.cos(ang[0]) * np.cos(ang[1]), -np.sin(ang[0]), np.cos(ang[0]) * np.sin(ang[1])],
#                  [np.sin(ang[0]) * np.cos(ang[1]), np.cos(ang[0]), np.sin(ang[0]) * np.sin(ang[1])],
#                  [-np.sin(ang[1]), 0, np.cos(ang[1])]]))
#        return rot_mat
#
#   
#    rot_matrix = rotation_matrix(orientation)
#    eigenvalue_matrix = np.array([[1 / semi_axis_a ** 2, 0, 0],
#                                  [0, 1 / semi_axis_a ** 2, 0],
#                                  [0, 0, 1 / semi_axis_c ** 2]])
#    E = np.dot(rot_matrix, np.dot(eigenvalue_matrix, np.transpose(rot_matrix)))
#    if plane is 'xy':
#        C = (np.array([[E[0,0], E[0,1]], [E[1,0], E[1,1]]]) 
#            - np.dot(np.transpose(np.array([E[2,0], E[2,1]])), np.dot(E[2,2], np.array(E[0,2], E[1,2]))))    
#    if plane is 'xz':
#        C = (np.array([[E[0,0], E[0,2]], [E[2,0], E[2,2]]]) 
#            - np.dot(np.transpose(np.array([E[1,0], E[2,1]])), np.dot(E[1,1], np.array(E[0,1], E[1,2]))))
#    if plane is 'yz':
#        C = (np.array([[E[1,1], E[1,2]], [E[2,1], E[2,2]]]) 
#            - np.dot(np.transpose(np.array([E[0,1], E[0,2]])), np.dot(E[0,0], np.array(E[1,0], E[2,0]))))
#        
#    
#    
#    
#    eigenval, eigenvec = np.linalg.eig(C)
#    semi_axis_a_prime = np.sqrt(1 / eigenval[0])
#    semi_axis_c_prime = np.sqrt(1 / eigenval[1])
#    alpha_prime = np.arccos(eigenvec[1,1])
#    
#    return semi_axis_a_prime, semi_axis_c_prime, alpha_prime
#
#    
#    