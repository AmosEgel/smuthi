"""Functions to call NFM-DS (null field method with discrete sources) Fortran code by Doicu et al. for the
generation of T-matrices for non-spherical particles.
The Fortran code comes with the book
A. Doicu, T. Wriedt, and Y. A. Eremin: Light Scattering by Systems of Particles,
1st ed. Berlin, Heidelberg: Springer-Verlag, 2006.
and can also be downloaded from
https://scattport.org/index.php/programs-menu/t-matrix-codes-menu/239-nfm-ds
"""

import smuthi.field_expansion as fldex
import smuthi.nfmds
import smuthi.memoizing as memo
import os
import subprocess
import numpy as np
import imp
import sys


@memo.Memoize
def tmatrix_spheroid(vacuum_wavelength=None, layer_refractive_index=None, particle_refractive_index=None,
                     semi_axis_c=None, semi_axis_a=None, l_max=None, m_max=None, use_ds=True, nint=None, nrank=None):
    """T-matrix for spheroid, using the TAXSYM.f90 routine from the NFM-DS.

    Args:
        vacuum_wavelength(float)
        layer_refractive_index(float):                  Real refractive index of layer (complex values are not allowed).
        particle_refractive_index(float or complex):    Complex refractive index of spheroid
        semi_axis_c (float):                            Semi axis of spheroid along rotation axis
        semi_axis_a (float):                            Semi axis of spheroid along lateral direction
        l_max (int):                                    Maximal multipole degree
        m_max (int):                                    Maximal multipole order
        use_ds (bool):                                  Flag to switch the use of discrete sources on (True) and
                                                        off (False)
        nint (int):                                     Nint parameter for internal use of NFM-DS (number of points
                                                        along integral). Higher value is more accurate and takes longer
        nrank (int):                                    l_max used internally in NFM-DS

    Returns:
        T-matrix as numpy.ndarray
    """
    filename = 'T_matrix_spheroid.dat'
    taxsym_write_input_spheroid(vacuum_wavelength=vacuum_wavelength, layer_refractive_index=layer_refractive_index,
                                particle_refractive_index=particle_refractive_index, semi_axis_c=semi_axis_c,
                                semi_axis_a=semi_axis_a, use_ds=use_ds, nint=nint, nrank=nrank, filename=filename)
    taxsym_run()
    return taxsym_read_tmatrix(filename=filename, l_max=l_max, m_max=m_max)


@memo.Memoize
def tmatrix_cylinder(vacuum_wavelength=None, layer_refractive_index=None, particle_refractive_index=None,
                     cylinder_height=None, cylinder_radius=None, l_max=None, m_max=None, use_ds=True, nint=None, nrank=None):
    """Return T-matrix for finite cylinder, using the TAXSYM.f90 routine from the NFM-DS.

    Args:
        vacuum_wavelength (float)
        layer_refractive_index (float):                 Real refractive index of layer (complex values are not allowed).
        particle_refractive_index (float or complex):   Complex refractive index of spheroid
        cylinder_height (float):                        Semi axis of spheroid along rotation axis
        cylinder_radius (float):                        Semi axis of spheroid along lateral direction
        l_max (int):                                    Maximal multipole degree
        m_max (int):                                    Maximal multipole order
        use_ds (bool):                                  Flag to switch the use of discrete sources on (True) and
                                                        off (False)
        nint (int):                                     Nint parameter for internal use of NFM-DS (number of points
                                                        along integral). Higher value is more accurate and takes longer
        nrank (int):                                    l_max used internally in NFM-DS

    Returns:
        T-matrix as numpy.ndarray
    """
    filename = 'T_matrix_cylinder.dat'
    taxsym_write_input_cylinder(vacuum_wavelength=vacuum_wavelength, layer_refractive_index=layer_refractive_index,
                                particle_refractive_index=particle_refractive_index, cylinder_height=cylinder_height,
                                cylinder_radius=cylinder_radius, use_ds=use_ds, nint=nint, nrank=nrank,
                                filename=filename)
    taxsym_run()
    return taxsym_read_tmatrix(filename=filename, l_max=l_max, m_max=m_max)


def taxsym_run():
    """Call TAXSYM.f90 routine."""
    smuthi.nfmds.initialize_binary()
    cwd = os.getcwd()
    os.chdir(smuthi.nfmds.nfmds_folder + '/TMATSOURCES')
    with open('../nfmds.log', 'w') as nfmds_log:
        if sys.platform.startswith('win'):
            subprocess.call(['TAXSYM_SMUTHI.exe'], stdout=nfmds_log)
        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            subprocess.call(['./TAXSYM_SMUTHI.out'], stdout=nfmds_log)
        else:
            raise AssertionError('Platform neither windows nor linux.')
    os.chdir(cwd)

    
def taxsym_write_input_spheroid(vacuum_wavelength=None, layer_refractive_index=None, particle_refractive_index=None,
                                semi_axis_c=None, semi_axis_a=None, use_ds=True, nint=None, nrank=None,
                                filename='T_matrix_spheroid.dat'):
    """Generate input file for the TAXSYM.f90 routine for the simulation of a spheroid.

    Args:
        vacuum_wavelength (float)
        layer_refractive_index (float):                 Real refractive index of layer (complex values are not allowed)
        particle_refractive_index (float or complex):   Complex refractive index of spheroid
        semi_axis_c (float):                            Semi axis of spheroid along rotation axis
        semi_axis_a (float):                            Semi axis of spheroid along lateral direction
        use_ds (bool):                                  Flag to switch the use of discrete sources on (True) and
                                                        off (False)
        nint (int):                                     Nint parameter for internal use of NFM-DS (number of points
                                                        along integral). Higher value is more accurate and takes longer
        nrank (int):                                    l_max used internally in NFM-DS
        filename (str):                                 Name of the file in which the T-matrix is stored
    """
    if layer_refractive_index.imag:
        raise ValueError('Refractive index of surrounding medium  must be real(?)')

    buffer = ('OptProp\n'
    + str(float(vacuum_wavelength)) + '\n'
    + str(float(layer_refractive_index.real)) + '\n'
    '(' + str((particle_refractive_index / layer_refractive_index).real) + ','
            + str((particle_refractive_index / layer_refractive_index).imag) + ')\n'

    'Variables:\n'
    ' - wavelength - wavelength of the incident light in vacuo.\n'
    ' - ind_refMed - refractive index of the ambient medium.\n'
    ' - ind_refRel - relative refractive index of the particle.  \n'

    '\n'
    'MatProp\n'
    '.false.\n'
    '.false.\n'
    '0.1\n'

    ' Variables:\n'
    ' - perfectcond - if perfectcond = t, the particle is perfectly conducting.\n'
    ' - chiral      - if chiral = t, the particle is optical active (chiral).\n'
    ' - kb          - parameter of chirality.\n'

    '\n'
    'GeomProp\n'
    '.false.\n'
    "'../GEOMFILES/prolate.fem'\n"
    '1\n'                          # TypeGeom = 1 for spheroid
    '2\n'                          # Nsurf = 2 for spheroid
    + str(float(semi_axis_c)) + '\n'        # half-height of spheroid
    + str(float(semi_axis_a)) + '\n'       # half-width of spheroid
    '1\n'                          # Nparam=1 for spheroid
    '1.0\n'
    '1.0\n'
    '.false.\n'

    ' Variables:\n'
    ' - FileGeom - if FileGeom = t, the particle geometry is supplied by the \n'
    '              input file FileFEM. \n'
    ' - FileFEM  - name of the file containing the particle geometry. \n'
    ' - TypeGeom - parameter specifying the type of the particle geometry.\n'
    ' - Nsurf	   - number of surface parameters. \n'
    ' - surf(1)  - surface parameter.\n'
    ' - ...  \n'
    ' - surf(Nsurf  \n'
    ' - Nparam   - number of smooth curves forming the generatrix curve.    \n'
    ' - anorm    - characteristic length of the particle which is used to \n'
    '              normalize the differential scattering cross sections.	 \n'
    ' - Rcirc    - characteristic length of the particle for computing Nrank. \n'
    ' - miror    - if miror = t, the particle is mirror symmetric.	            \n'
    ' NOTE: FOR CHIRAL PARTICLES AND DISTRIBUTED SOURCES SET miror = f.\n'

    '\n'
    'ConvTest\n'
    '.false.\n'
    '.false.\n'

    ' Variables:\n'
    ' - DoConvTest   - if DoConvTest = t, the interactive convergence tests \n'
    '                  over Nint and Nrank are performed.   \n'
    ' - MishConvTest - if MishConvTest = t, estimates of Nint and Nrank are  \n'
    '                  computed with the convergence criterion proposed by \n'
    '                  Mishchenko.        \n'
    ' NOTE: IF THE PARTICLE IS OPTICAL ACTIVE (chiral = t) OR THE PARTICLE\n'
    ' GEOMETRY IS SUPPLIED BY THE FILE FileFEM (FileGeom = t), THE CODE SETS\n'
    ' MishConvTest = f. IN FACT, MISHCHENKO''S CONVERGENCE TEST WILL BE \n'
    ' PERFORMED IF (DS = f AND DoConvTest = t AND chiral = f AND FileGeom = f), \n'
    ' OR (DS = t AND autGenDS = t AND DoConvTest = t AND chiral = f AND \n'
    ' FileGeom = f).   \n'

    '\n'
    'Sources\n')
    if use_ds:
        buffer += '.true.\n'
    else:
        buffer += '.false.\n'
    buffer += ('.true.\n'

    ' Variables:\n'
    ' - DS       - if DS = t, distributed sources are used for T-matrix \n'
    '              calculation. 	\n'
    ' - autGenDS - if autGenDS = t, the coordinates of the distributed sources\n'
    '              are generated by the code.\n'
    ' NOTE: IF THE PARTICLE GEOMETRY IS READ FROM FILE (FileGeom = t),\n'
    ' THE CODE SETS autgenDS = f.                                 \n'

    '\n'
    'SourcePosAut\n'
    '.true.\n'
    '0.95\n'

    ' Variables: \n'
    ' - ComplexPlane - if ComplexPlane = t, the distributed sources are placed\n'
    '                  in the complex plane.\n'
    ' - EpsZReIm     - parameter controlling the distribution of the discrete \n'
    '                  sources.\n'
    ' NOTE: THESE VARIABLES MUST BE PROVIDED IF (DS = t AND autgenDS = t).\n'

    '\n'
    'NintNrank\n'
    + str(nint) + '\n'
    + str(nrank) + '\n'

    ' Variables: \n'
    ' - Nint  - number of integration points in computing integrals over the \n'
    '           generatrix curve.\n'
    ' - Nrank - maximum expansion order.  \n'
    ' NOTE: THESE VARIABLES MUST BE PROVIDED IF ((DoConvTest = f) OR \n'
    ' (DS = t AND autgenDS = f)).                  \n'

    '\n'
    'Errors\n'
    '5.e-2\n'
    '5.e-2\n'
    '1.e-2\n'
    '4\n'
    '50\n'
    ' Variables:\n'
    ' - epsNint    - error tolerance for the integration test.    \n'
    ' - epsNrank   - error tolerance for the expansion order test.  \n'
    ' - epsMrank   - error tolerance for the azimuthal order test.  \n'
    ' - dNint	     - number of division points for the integration test \n'
    '                and Mishchenko''s convergence test.   \n'
    ' - dNintMrank - number of division points for azimuthal mode \n'
    '                calculation.\n'

    '\n'
    'Tmat\n'
    "'../TMATFILES/" + filename + "'\n"
    ' Variable:\n'
    ' - FileTmat - name of the file to which the T matrix is written.  \n'

    '\n'
    'PrintProgress\n'
    '.false.\n'
    ' Variable:\n'
    ' - PrnProgress - if PrnProgress = t, the progress of calculation \n'
    '                 is printed. \n'

    ' Comment\n'
    ' This file was generated by the routine smuthi.nfmds_wrappers.taxsym_write_input_spheroid \n')

    smuthi.nfmds.initialize_source()
    f = open(smuthi.nfmds.nfmds_folder + '/INPUTFILES/InputAXSYM.dat', 'w')
    f.write(buffer)
    f.close()


def taxsym_write_input_cylinder(vacuum_wavelength=None, layer_refractive_index=None, particle_refractive_index=None,
                                cylinder_height=None, cylinder_radius=None, use_ds=True, nint=None, nrank=None,
                                filename='T_matrix_cylinder.dat'):
    """Generate input file for the TAXSYM.f90 routine for the simulation of a finite cylinder.

    Args:
        vacuum_wavelength (float)
        layer_refractive_index (float):                 Real refractive index of layer (complex values are not allowed)
        particle_refractive_index (float or complex):   Complex refractive index of cylinder
        cylinder_height (float):                        Height of cylinder (length unit)
        cylinder_radius (float):                        Radius of cylinder (length unit)
        use_ds (bool):                                  Flag to switch the use of discrete sources on (True) and
                                                        off (False)
        nint (int):                                     Nint parameter for internal use of NFM-DS (number of points
                                                        along integral). Higher value is more accurate and takes longer
        nrank (int):                                    l_max used internally in NFM-DS
        filename (str):                                 Name of the file in which the T-matrix is stored
    """
    if layer_refractive_index.imag:
        raise ValueError('Refractive index of surrounding medium  must be real(?)')

    buffer = ('OptProp\n'
    + str(float(vacuum_wavelength)) + '\n'
    + str(float(layer_refractive_index.real)) + '\n'
    '(' + str((particle_refractive_index / layer_refractive_index).real) + ','
            + str((particle_refractive_index / layer_refractive_index).imag) + ')\n'

    'Variables:\n'
    ' - wavelength - wavelength of the incident light in vacuo.\n'
    ' - ind_refMed - refractive index of the ambient medium.\n'
    ' - ind_refRel - relative refractive index of the particle.  \n'

    '\n'
    'MatProp\n'
    '.false.\n'
    '.false.\n'
    '0.1\n'

    ' Variables:\n'
    ' - perfectcond - if perfectcond = t, the particle is perfectly conducting.\n'
    ' - chiral      - if chiral = t, the particle is optical active chiral.\n'
    ' - kb          - parameter of chirality.\n'

    '\n'
    'GeomProp\n'
    '.false.\n'
    "'../GEOMFILES/prolate.fem'\n"
    '2\n'                          # TypeGeom=2 for cylinder
    '2\n'                          # Nsurf=2 for cylinder
    + str(float(cylinder_height / 2)) + '\n'        # half-height of cylinder
    + str(float(cylinder_radius)) + '\n'       # radius of cylinder
    '3\n'                          # Nparam=3 for cylinder
    '1.0\n'
    '1.0\n'
    '.false.\n'

    ' Variables:\n'
    ' - FileGeom - if FileGeom = t, the particle geometry is supplied by the \n'
    '              input file FileFEM. \n'
    ' - FileFEM  - name of the file containing the particle geometry. \n'
    ' - TypeGeom - parameter specifying the type of the particle geometry.\n'
    ' - Nsurf	   - number of surface parameters. \n'
    ' - surf1  - surface parameter.\n'
    ' - ...  \n'
    ' - surfNsurf  \n'
    ' - Nparam   - number of smooth curves forming the generatrix curve.    \n'
    ' - anorm    - characteristic length of the particle which is used to \n'
    '              normalize the differential scattering cross sections.	 \n'
    ' - Rcirc    - characteristic length of the particle for computing Nrank. \n'
    ' - miror    - if miror = t, the particle is mirror symmetric.	            \n'
    ' NOTE: FOR CHIRAL PARTICLES AND DISTRIBUTED SOURCES SET miror = f.\n'

    '\n'
    'ConvTest\n'
    '.false.\n'
    '.false.\n'

    ' Variables:\n'
    ' - DoConvTest   - if DoConvTest = t, the interactive convergence tests \n'
    '                  over Nint and Nrank are performed.   \n'
    ' - MishConvTest - if MishConvTest = t, estimates of Nint and Nrank are  \n'
    '                  computed with the convergence criterion proposed by \n'
    '                  Mishchenko.        \n'
    ' NOTE: IF THE PARTICLE IS OPTICAL ACTIVE (chiral = t) OR THE PARTICLE\n'
    ' GEOMETRY IS SUPPLIED BY THE FILE FileFEM (FileGeom = t), THE CODE SETS\n'
    ' MishConvTest = f. IN FACT, MISHCHENKO''S CONVERGENCE TEST WILL BE \n'
    ' PERFORMED IF (DS = f AND DoConvTest = t AND chiral = f AND FileGeom = f), \n'
    ' OR (DS = t AND autGenDS = t AND DoConvTest = t AND chiral = f AND \n'
    ' FileGeom = f).   \n'

    '\n'
    'Sources\n')
    if use_ds:
        buffer += '.true.\n'
    else:
        buffer += '.false.\n'
    buffer += ('.true.\n'

    ' Variables:\n'
    ' - DS       - if DS = t, distributed sources are used for T-matrix \n'
    '              calculation. 	\n'
    ' - autGenDS - if autGenDS = t, the coordinates of the distributed sources\n'
    '              are generated by the code.\n'
    ' NOTE: IF THE PARTICLE GEOMETRY IS READ FROM FILE (FileGeom = t),\n'
    ' THE CODE SETS autgenDS = f.                                 \n'

    '\n'
    'SourcePosAut\n'
    '.true.\n'
    '0.95\n'

    ' Variables: \n'
    ' - ComplexPlane - if ComplexPlane = t, the distributed sources are placed\n'
    '                  in the complex plane.\n'
    ' - EpsZReIm     - parameter controlling the distribution of the discrete \n'
    '                  sources.\n'
    ' NOTE: THESE VARIABLES MUST BE PROVIDED IF (DS = t AND autgenDS = t).\n'

    '\n'
    'NintNrank\n'
    + str(nint) + '\n'
    + str(nrank) + '\n'

    ' Variables: \n'
    ' - Nint  - number of integration points in computing integrals over the \n'
    '           generatrix curve.\n'
    ' - Nrank - maximum expansion order.  \n'
    ' NOTE: THESE VARIABLES MUST BE PROVIDED IF ((DoConvTest = f) OR \n'
    ' (DS = t AND autgenDS = f)).                  \n'

    '\n'
    'Errors\n'
    '5.e-2\n'
    '5.e-2\n'
    '1.e-2\n'
    '4\n'
    '50\n'
#     '1.e-2\n'
#     '1.e-2\n'
#     '5.e-3\n'
#     '8\n'
#     '100\n'
    ' Variables:\n'
    ' - epsNint    - error tolerance for the integration test.    \n'
    ' - epsNrank   - error tolerance for the expansion order test.  \n'
    ' - epsMrank   - error tolerance for the azimuthal order test.  \n'
    ' - dNint	     - number of division points for the integration test \n'
    '                and Mishchenko''s convergence test.   \n'
    ' - dNintMrank - number of division points for azimuthal mode \n'
    '                calculation.\n'

    '\n'
    'Tmat\n'
    "'../TMATFILES/" + filename + "'\n"
    ' Variable:\n'
    ' - FileTmat - name of the file to which the T matrix is written.  \n'

    '\n'
    'PrintProgress\n'
    '.false.\n'
    ' Variable:\n'
    ' - PrnProgress - if PrnProgress = t, the progress of calculation \n'
    '                 is printed. \n'

    ' Comment\n'
    ' This file was generated by the routine smuthi.nfmds_wrappers.taxsym_write_input_cylinder \n')

    smuthi.nfmds.initialize_source()
    f = open(smuthi.nfmds.nfmds_folder + '/INPUTFILES/InputAXSYM.dat', 'w')
    f.write(buffer)
    f.close()


def taxsym_read_tmatrix(filename, l_max, m_max):
    """Export TAXSYM.f90 output to SMUTHI T-matrix.

    .. todo:: feedback to adapt particle m_max to nfmds m_max

    Args:
        filename (str): Name of the file containing the T-matrix output of TAXSYM.f90
        l_max (int):    Maximal multipole degree
        m_max (int):    Maximal multipole order

    Returns:
        T-matrix as numpy.ndarray
    """

    with open(smuthi.nfmds.nfmds_folder + '/TMATFILES/Info' + filename, 'r') as info_file:
        info_file_lines = info_file.readlines()

    assert 'The scatterer is an axisymmetric particle' in ' '.join(info_file_lines)

    for line in info_file_lines:
        if line.split()[0:4] == ['-', 'maximum', 'expansion', 'order,']:
            n_rank = int(line.split()[-1][0:-1])

        if line.split()[0:5] == ['-', 'number', 'of', 'azimuthal', 'modes,']:
            m_rank = int(line.split()[-1][0:-1])

    with open(smuthi.nfmds.nfmds_folder + '/TMATFILES/' + filename, 'r') as tmat_file:
        tmat_lines = tmat_file.readlines()

    t_nfmds = [[]]
    column_index = 0
    for line in tmat_lines[3:]:
        split_line = line.split()
        for i_entry in range(int(len(split_line) / 2)):
            if column_index == 2 * n_rank:
                t_nfmds.append([])
                column_index = 0
            t_nfmds[-1].append(complex(split_line[2 * i_entry]) + 1j * complex(split_line[2 * i_entry + 1]))
            column_index += 1

    t_matrix = np.zeros((fldex.blocksize(l_max, m_max), fldex.blocksize(l_max, m_max)), dtype=complex)

    for m in range(-m_max, m_max + 1):
        n_max_nfmds = n_rank - max(1, abs(m)) + 1
        for tau1 in range(2):
            for l1 in range(max(1, abs(m)), l_max + 1):
                n1 = fldex.multi_to_single_index(tau=tau1, l=l1, m=m, l_max=l_max, m_max=m_max)
                l1_nfmds = l1 - max(1, abs(m))
                n1_nfmds = 2 * n_rank * abs(m) + tau1 * n_max_nfmds + l1_nfmds
                for tau2 in range(2):
                    for l2 in range(max(1, abs(m)), l_max + 1):
                        n2 = fldex.multi_to_single_index(tau=tau2, l=l2, m=m, l_max=l_max, m_max=m_max)
                        l2_nfmds = l2 - max(1, abs(m))
                        n2_nfmds = tau2 * n_max_nfmds + l2_nfmds
                        if abs(m) <= m_rank:
                            if m >= 0:
                                t_matrix[n1, n2] = t_nfmds[n1_nfmds][n2_nfmds]
                            else:
                                t_matrix[n1, n2] = t_nfmds[n1_nfmds][n2_nfmds] * (-1) ** (tau1 + tau2)

    return t_matrix
