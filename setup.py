# -*- coding: utf-8 -*-
import os
from setuptools import setup
from setuptools.command.install import install
import pkg_resources
import os
import subprocess
import sys


class CustomInstallCommand(install):
    """Compile nfmds code."""
    def run(self):
        install.run(self)
        # compile nfmds if not called by sphinx autodoc
        if ((sys.platform.startswith('linux') or sys.platform.startswith('darwin'))
            and not 'sphinx' in sys.modules):
            nfmds_sources_dirname = pkg_resources.resource_filename('smuthi.nfmds', 'NFM-DS')
            os.chdir(nfmds_sources_dirname + '/TMATSOURCES')
            sys.stdout.write('Compiling sources at ' + nfmds_sources_dirname + ' ...')
            sys.stdout.flush()
            subprocess.call(['gfortran', 'TAXSYM_SMUTHI.f90', '-o', 'TAXSYM_SMUTHI.out'])
            sys.stdout.write(' done.\n')
            sys.stdout.flush()


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="SMUTHI",
    version="0.8.4",
    author="Amos Egel",
    author_email="amos.egel@kit.edu",
    url='https://gitlab.com/AmosEgel/smuthi',
    description="Light scattering by multiple particles in thin-film systems",
    long_description=read('README.rst'),
    packages=['smuthi', 'smuthi.nfmds'],
    cmdclass={'install': CustomInstallCommand},
    package_data={'smuthi.nfmds': ['NFM-DS/*.txt', 'NFM-DS/TMATSOURCES/*', 'NFM-DS/TMATFILES/*',
                                   'NFM-DS/INPUTFILES/*.dat', 'NFM-DS/OUTPUTFILES/*'],
                  'smuthi': ['data/*']},
    install_requires=['numpy', 'scipy', 'mpmath', 'matplotlib', 'pyyaml', 'argparse', 'imageio', 'sympy', 'tqdm'],
    extras_require={'cuda':  ['PyCuda']},
    entry_points={'console_scripts': ['smuthi = smuthi.__main__:main']},
    license='MIT',
)
