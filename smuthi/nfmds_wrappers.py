"""Functions to call NFM-DS (null field method with discrete sources) Fortran code by Doicu et al. for the
generation of T-matrices for non-spherical particles."""

import smuthi.index_conversion as idx
import os
from subprocess import Popen, PIPE
import numpy as np
import sys


def taxsym_tmatrix_spheroid(vacuum_wavelength=None, layer_refractive_index=None, particle_refractive_index=None,
                            half_axis_z=None, half_axis_xy=None, use_ds=True, n_int=None, n_rank=None):

    filename = 'T_matrix_spheroid.dat'

    write_taxsym_input_spheroid(vacuum_wavelength=vacuum_wavelength, layer_refractive_index=layer_refractive_index,
                                particle_refractive_index=particle_refractive_index, half_axis_z=half_axis_z,
                                half_axis_xy=half_axis_xy, use_ds=use_ds, n_int=n_int, n_rank=n_rank,
                                filename=filename)
    run_taxsym()
    t_matrix = read_taxsym_tmatrix(filename=filename)
    return t_matrix


def run_taxsym():
    nfmds_log = open('nfmds.log', 'a')
    os.chdir('NFM-DS/TMATSOURCES')
    p = Popen('main.exe', stdin=PIPE, stdout=nfmds_log, universal_newlines=True)
    p.communicate('1')
    os.chdir('../..')
    nfmds_log.close()


def write_taxsym_input_spheroid(vacuum_wavelength=None, layer_refractive_index=None, particle_refractive_index=None,
                                half_axis_z=None, half_axis_xy=None, use_ds=True, n_int=None, n_rank=None,
                                filename='T_matrix_spheroid.dat'):

    if layer_refractive_index.imag:
        raise ValueError('Refractive index of surrounding medium  must be real(?)')

    f = open('NFM-DS/INPUTFILES/InputAXSYM.dat', 'w')

    f.write('OptProp\n')
    f.write(str(float(vacuum_wavelength)) + '\n')
    f.write(str(float(layer_refractive_index.real)) + '\n')
    f.write('(' + str((particle_refractive_index / layer_refractive_index).real) + ','
            + str((particle_refractive_index / layer_refractive_index).imag) + ')\n')

    f.write('Variables:\n')
    f.write(' - wavelength - wavelength of the incident light in vacuo.\n')
    f.write(' - ind_refMed - refractive index of the ambient medium.\n')
    f.write(' - ind_refRel - relative refractive index of the particle.  \n')

    f.write('\n')
    f.write('MatProp\n')
    f.write('.false.\n')
    f.write('.false.\n')
    f.write('0.1\n')

    f.write(' Variables:\n')
    f.write(' - perfectcond - if perfectcond = t, the particle is perfectly conducting.\n')
    f.write(' - chiral      - if chiral = t, the particle is optical active (chiral).\n')
    f.write(' - kb          - parameter of chirality.\n')

    f.write('\n')
    f.write('GeomProp\n')
    f.write('.false.\n')
    f.write("'../GEOMFILES/prolate.fem'\n")
    f.write('1\n')                          # TypeGeom = 1 for spheroid
    f.write('2\n')                          # Nsurf = 2 for spheroid
    f.write(str(float(half_axis_z)) + '\n')        # half-height of spheroid
    f.write(str(float(half_axis_xy)) + '\n')       # half-width of spheroid
    f.write('1\n')                          # Nparam=1 for spheroid
    f.write('1.0\n')
    f.write('1.0\n')
    f.write('.false.\n')

    f.write(' Variables:\n')
    f.write(' - FileGeom - if FileGeom = t, the particle geometry is supplied by the \n')
    f.write('              input file FileFEM. \n')
    f.write(' - FileFEM  - name of the file containing the particle geometry. \n')
    f.write(' - TypeGeom - parameter specifying the type of the particle geometry.\n')
    f.write(' - Nsurf	   - number of surface parameters. \n')
    f.write(' - surf(1)  - surface parameter.\n')
    f.write(' - ...  \n')
    f.write(' - surf(Nsurf)  \n')
    f.write(' - Nparam   - number of smooth curves forming the generatrix curve.    \n')
    f.write(' - anorm    - characteristic length of the particle which is used to \n')
    f.write('              normalize the differential scattering cross sections.	 \n')
    f.write(' - Rcirc    - characteristic length of the particle for computing Nrank. \n')
    f.write(' - miror    - if miror = t, the particle is mirror symmetric.	            \n')
    f.write(' NOTE: FOR CHIRAL PARTICLES AND DISTRIBUTED SOURCES SET miror = f.\n')

    f.write('\n')
    f.write('ConvTest\n')
    f.write('.false.\n')
    f.write('.false.\n')

    f.write(' Variables:\n')
    f.write(' - DoConvTest   - if DoConvTest = t, the interactive convergence tests \n')
    f.write('                  over Nint and Nrank are performed.   \n')
    f.write(' - MishConvTest - if MishConvTest = t, estimates of Nint and Nrank are  \n')
    f.write('                  computed with the convergence criterion proposed by \n')
    f.write('                  Mishchenko.        \n')
    f.write(' NOTE: IF THE PARTICLE IS OPTICAL ACTIVE (chiral = t) OR THE PARTICLE\n')
    f.write(' GEOMETRY IS SUPPLIED BY THE FILE FileFEM (FileGeom = t), THE CODE SETS\n')
    f.write(' MishConvTest = f. IN FACT, MISHCHENKO''S CONVERGENCE TEST WILL BE \n')
    f.write(' PERFORMED IF (DS = f AND DoConvTest = t AND chiral = f AND FileGeom = f), \n')
    f.write(' OR (DS = t AND autGenDS = t AND DoConvTest = t AND chiral = f AND \n')
    f.write(' FileGeom = f).   \n')

    f.write('\n')
    f.write('Sources\n')
    if use_ds:
        f.write('.true.\n')
    else:
        f.write('.false.\n')
    f.write('.true.\n')

    f.write(' Variables:\n')
    f.write(' - DS       - if DS = t, distributed sources are used for T-matrix \n')
    f.write('              calculation. 	\n')
    f.write(' - autGenDS - if autGenDS = t, the coordinates of the distributed sources\n')
    f.write('              are generated by the code.\n')
    f.write(' NOTE: IF THE PARTICLE GEOMETRY IS READ FROM FILE (FileGeom = t),\n')
    f.write(' THE CODE SETS autgenDS = f.                                 \n')

    f.write('\n')
    f.write('SourcePosAut\n')
    f.write('.true.\n')
    f.write('0.95\n')

    f.write(' Variables: \n')
    f.write(' - ComplexPlane - if ComplexPlane = t, the distributed sources are placed\n')
    f.write('                  in the complex plane.\n')
    f.write(' - EpsZReIm     - parameter controlling the distribution of the discrete \n')
    f.write('                  sources.\n')
    f.write(' NOTE: THESE VARIABLES MUST BE PROVIDED IF (DS = t AND autgenDS = t).\n')

    f.write('\n')
    f.write('NintNrank\n')
    f.write(str(n_int) + '\n')
    f.write(str(n_rank) + '\n')

    f.write(' Variables: \n')
    f.write(' - Nint  - number of integration points in computing integrals over the \n')
    f.write('           generatrix curve.\n')
    f.write(' - Nrank - maximum expansion order.  \n')
    f.write(' NOTE: THESE VARIABLES MUST BE PROVIDED IF ((DoConvTest = f) OR \n')
    f.write(' (DS = t AND autgenDS = f)).                  \n')

    f.write('\n')
    f.write('Errors\n')
    f.write('5.e-2\n')
    f.write('5.e-2\n')
    f.write('1.e-2\n')
    f.write('4\n')
    f.write('50\n')
    f.write(' Variables:\n')
    f.write(' - epsNint    - error tolerance for the integration test.    \n')
    f.write(' - epsNrank   - error tolerance for the expansion order test.  \n')
    f.write(' - epsMrank   - error tolerance for the azimuthal order test.  \n')
    f.write(' - dNint	     - number of division points for the integration test \n')
    f.write('                and Mishchenko''s convergence test.   \n')
    f.write(' - dNintMrank - number of division points for azimuthal mode \n')
    f.write('                calculation.\n')

    f.write('\n')
    f.write('Tmat\n')
    f.write("'../TMATFILES/" + filename + "'\n")
    f.write(' Variable:\n')
    f.write(' - FileTmat - name of the file to which the T matrix is written.  \n')

    f.write('\n')
    f.write('PrintProgress\n')
    f.write('.true.\n')
    f.write(' Variable:\n')
    f.write(' - PrnProgress - if PrnProgress = t, the progress of calculation \n')
    f.write('                 is printed. \n')

    f.write(' Comment\n')
    f.write(' This file was generated by the routine smuthi.nfmds_wrappers.write_taxsym_input_spheroid \n')

    f.close()


def read_taxsym_tmatrix(filename):

    info_file = open('NFM-DS/TMATFILES/Info' + filename, 'r')
    info_file_lines = info_file.readlines()
    info_file.close()

    assert 'The scatterer is an axisymmetric particle' in ' '.join(info_file_lines)

    for line in info_file_lines:
        if line.split()[0:4] == ['-', 'maximum', 'expansion', 'order,']:
            n_rank = int(line.split()[-1][0:-1])

        if line.split()[0:5] == ['-', 'number', 'of', 'azimuthal', 'modes,']:
            m_rank = int(line.split()[-1][0:-1])

    tmat_file = open('NFM-DS/TMATFILES/' + filename, 'r')
    tmat_lines = tmat_file.readlines()
    tmat_file.close()

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

    t_matrix = np.zeros((idx.number_of_indices(), idx.number_of_indices()), dtype=complex)

    for m in range(-idx.l_max, idx.l_max + 1):
        n_max_nfmds = n_rank - max(1, abs(m)) + 1
        for tau1 in range(2):
            for l1 in range(max(1, abs(m)), idx.l_max + 1):
                n1 = idx.multi_to_single_index(tau=tau1, l=l1, m=m)
                l1_nfmds = l1 - max(1, abs(m))
                n1_nfmds = 2 * n_rank * abs(m) + tau1 * n_max_nfmds + l1_nfmds
                for tau2 in range(2):
                    for l2 in range(max(1, abs(m)), idx.l_max + 1):
                        n2 = idx.multi_to_single_index(tau=tau2, l=l2, m=m)
                        l2_nfmds = l2 - max(1, abs(m))
                        n2_nfmds = tau2 * n_max_nfmds + l2_nfmds
                        if abs(m) <= m_rank:
                            if m >= 0:
                                t_matrix[n1, n2] = t_nfmds[n1_nfmds][n2_nfmds]
                            else:
                                t_matrix[n1, n2] = t_nfmds[n1_nfmds][n2_nfmds] * (-1) ** (tau1 + tau2)

    return t_matrix