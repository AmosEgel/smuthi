"""Check if NFM-DS is installed and otherwise do so."""
import pkg_resources
import shutil
import os
import subprocess
import sys


nfmds_sources_dirname = pkg_resources.resource_filename('smuthi.nfmds', 'NFM-DS')
with open(nfmds_sources_dirname + '/_nfmds_installation_path.txt', 'r') as rfile:
    nfmds_installation_path = rfile.read()


def install_nfmds():
    global nmfds_installation_path
    nfmds_installation_path = os.path.abspath(input('Please type a path where NFM-DS will be installed!\n') + '/NFM-DS')
    shutil.copytree(nfmds_sources_dirname, nfmds_installation_path)
    try:
        with open(nfmds_sources_dirname + '/_nfmds_installation_path.txt', 'w') as wfile:
            wfile.write(nfmds_installation_path)
    except:
        print('Permission denied: Could not write to ' + nfmds_sources_dirname + '/_nfmds_installation_path.txt.')
        print('Consider to run once with admin rights to keep NFM-DS installation.')
    if sys.platform.startswith('linux'):
        cwd = os.getcwd()
        os.chdir(nfmds_installation_path + '/TMATSOURCES')
        print('Compiling sources ...')
        subprocess.call(['gfortran', 'TAXSYM_SMUTHI.f90', '-o', 'TAXSYM_SMUTHI.out'])
        print('... done.')
        os.chdir(cwd)
