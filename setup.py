# -*- coding: utf-8 -*-
"""This module is needed for the installation of the package."""

import os
from setuptools import setup
from setuptools.command.install import install
import pkg_resources
import os
import subprocess
import sys
import warnings

version = {}
with open("smuthi/version.py") as fp:
    exec(fp.read(), version)
__version__ = version['__version__']


class CustomInstallCommand(install):
    """Compile nfmds code."""
    def run(self):
        install.run(self)
        # compile nfmds if not built on readthedocs
        if ((sys.platform.startswith('linux') or sys.platform.startswith('darwin'))
            and not os.environ.get('READTHEDOCS')):
            nfmds_sources_dirname = pkg_resources.resource_filename('smuthi.linearsystem.tmatrix.nfmds', 'NFM-DS')
            os.chdir(nfmds_sources_dirname + '/TMATSOURCES')
            sys.stdout.write('Compiling sources at ' + nfmds_sources_dirname + ' ...')
            sys.stdout.flush()
            subprocess.call(['gfortran', 'TAXSYM_SMUTHI.f90', '-o', 'TAXSYM_SMUTHI.out'])
            sys.stdout.write(' done.\n')
            sys.stdout.flush()


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_requirements():
    requirements = ['argparse',
                    'imageio',
                    'matplotlib',
                    'mpmath',
                    'numpy',
                    'numba',
                    'pyyaml',
                    'scipy',
                    'sympy',
                    'tqdm',]
    if sys.platform.startswith('win'):
        warnings.warn('\n****************************************************\n'
                      'Due to reported issues, the installation of pywigxjpf is omitted on Windows machines.\n'
                      'If you want to benefit from faster evaluation of Wigner-3j symbols,\n'
                      'try to manually install that package, e.g. by "pip install pywigxjpf".'
                      '\n****************************************************\n',                      
                      UserWarning)
    else:
        requirements.append('pywigxjpf')
    return requirements

setup(
    name="SMUTHI",
    version=__version__,
    author="Amos Egel",
    author_email="amos.egel@gmail.com",
    url='https://gitlab.com/AmosEgel/smuthi',
    description="Light scattering by multiple particles in thin-film systems",
    long_description=read('README.rst'),
    packages=['smuthi',
              'smuthi.fields',
              'smuthi.linearsystem',
              'smuthi.linearsystem.tmatrix',
              'smuthi.linearsystem.tmatrix.nfmds',
              'smuthi.linearsystem.particlecoupling',
              'smuthi.postprocessing',
              'smuthi.utility'],
    cmdclass={'install': CustomInstallCommand},
    package_data={'smuthi.linearsystem.tmatrix.nfmds': ['NFM-DS/*.txt', 'NFM-DS/TMATSOURCES/*', 'NFM-DS/TMATFILES/*',
                                                        'NFM-DS/INPUTFILES/*.dat', 'NFM-DS/OUTPUTFILES/*'],
                  'smuthi': ['_data/*']},
    install_requires=get_requirements(),
    extras_require={'cuda':  ['PyCuda']},
    entry_points={'console_scripts': ['smuthi = smuthi.__main__:main']},
    license='MIT',
)
