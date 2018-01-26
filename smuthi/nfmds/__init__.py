"""Check if NFM-DS is installed and otherwise do so."""
import pkg_resources
import os
import subprocess
import sys
from distutils.dir_util import copy_tree
import tempfile

cwd = os.getcwd()
bindir = os.path.join(cwd,'smuthi_nfmds_bin_tmp')

nfmds_sources_dirname = pkg_resources.resource_filename('smuthi.nfmds', 'NFM-DS')
try:
    temp_fold = tempfile.TemporaryDirectory(prefix='smuthi_nfmds_')
except:
    sys.stdout.write('Faild to create temporary directory ... Using '+bindir+' instead\n')
    sys.stdout.flush()    
    temp_fold = bindir
    

nfmds_temporary_folder = temp_fold.name

if os.path.exists(bindir):
    nfmds_temporary_folder = bindir

nfmds_files = ['_README.txt', 'OUTPUTFILES', 'GEOMFILES', 'TMATFILES', 'TMATSOURCES', 'INPUTFILES']    
if len(  set(os.listdir(nfmds_temporary_folder)) & set(nfmds_files) ) != 6:  
    sys.stdout.write('Wrong number of files in NFMDS directory '+nfmds_temporary_folder+'\nCopying NFMDS files ...\n')
    sys.stdout.flush()
    copy_tree(nfmds_sources_dirname, nfmds_temporary_folder)

if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
    # does the executable exist?
    if not os.access(nfmds_temporary_folder + '/TMATSOURCES/TAXSYM_SMUTHI.out', os.X_OK):
        cwd = os.getcwd()
        os.chdir(nfmds_temporary_folder + '/TMATSOURCES')
        sys.stdout.write('Compiling sources ...')
        sys.stdout.flush()
        subprocess.call(['gfortran', 'TAXSYM_SMUTHI.f90', '-o', 'TAXSYM_SMUTHI.out'])
        sys.stdout.write(' done.\n')
        sys.stdout.flush()
        os.chdir(cwd)
        
