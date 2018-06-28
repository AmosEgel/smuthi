"""Check if NFM-DS is installed and otherwise do so."""
import pkg_resources
import os
import subprocess
import sys
from distutils.dir_util import copy_tree
import tempfile
try:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
except:
    mpi_rank = 0


cwd = os.getcwd()
cwd_bindir = os.path.join(cwd,'smuthi_nfmds_bin')

# set the folder for the (temporary) NFM-DS installation
if os.path.exists(cwd_bindir):
    nfmds_folder = cwd_bindir
else:
    temp_fold = tempfile.TemporaryDirectory(prefix='smuthi_nfmds_')
    nfmds_folder = temp_fold.name

if mpi_rank != 0:
    mpi_bindir = os.path.join(nfmds_folder, str(mpi_rank))
    if not os.path.exists(mpi_bindir):
        os.mkdir(mpi_bindir)
    nfmds_folder = mpi_bindir

nfmds_sources_dirname = pkg_resources.resource_filename('smuthi.nfmds', 'NFM-DS')

# check if the required folders are there, otherwise copy
nfmds_files = ['_README.txt', 'OUTPUTFILES', 'GEOMFILES', 'TMATFILES', 'TMATSOURCES', 'INPUTFILES']
if len(set(os.listdir(nfmds_folder)) & set(nfmds_files)) != 6:
    sys.stdout.write('Copying NFMDS files to ' + nfmds_folder + '\n')
    sys.stdout.flush()
    copy_tree(nfmds_sources_dirname, nfmds_folder)

# check if executable exists, otherwise compile if not built on readthedocs
if ((sys.platform.startswith('linux') or sys.platform.startswith('darwin'))
    and not os.access(nfmds_folder + '/TMATSOURCES/TAXSYM_SMUTHI.out', os.X_OK)
    and not os.environ.get('READTHEDOCS')):
    cwd = os.getcwd()
    os.chdir(nfmds_folder + '/TMATSOURCES')
    sys.stdout.write('Compiling sources ...')
    sys.stdout.flush()
    subprocess.call(['gfortran', 'TAXSYM_SMUTHI.f90', '-o', 'TAXSYM_SMUTHI.out'])
    sys.stdout.write(' done.\n')
    sys.stdout.flush()
    os.chdir(cwd)
