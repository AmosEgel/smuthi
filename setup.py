# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.command.install import install
import os
import sys
import subprocess

# chdir to smuthi project folder
smuthi_folder_path = os.path.normpath(os.path.dirname(os.path.realpath(__file__)))
original_path = os.getcwd()
os.chdir(smuthi_folder_path)

with open('README.rst', 'r') as readme:
    README_TEXT = readme.read()


class CustomInstallCommand(install):
    """Customized setuptools install command - compiles TAXSYM.f90."""
    def run(self):
        os.chdir('NFM-DS/TMATSOURCES')
        try:
            if sys.platform.startswith('win'):
                subprocess.call(['gfortran', 'TAXSYM.f90', '-o',  'taxsym.exe'])
                print('successfully compiled TAXSYM.f90')
            elif sys.platform.startswith('linux'):
                subprocess.call(['gfortran', 'TAXSYM.f90', '-o', 'taxsym.out'])
                print('successfully compiled TAXSYM.f90')
            else:
                raise AssertionError('Platform neither windows nor linux.')
        except:
            print('failed to compile TAXSYM.f90')
        os.chdir('../..')
        install.run(self)

setup(
    name="SMUTHI",
    version="0.1.2",
    author="Amos Egel",
    author_email="amos.egel@kit.edu",
    url='https://gitlab.com/AmosEgel/smuthi',
    description="Light scattering by multiple particles in thin-film systems",
    long_description=README_TEXT,
    packages=['smuthi'],
    install_requires=['numpy', 'scipy', 'sympy', 'matplotlib', 'pyyaml', 'argparse'],
    entry_points={'console_scripts': ['smuthi = smuthi.__main__:main']},
    license='MIT',
    cmdclass={'install': CustomInstallCommand}
)

os.chdir(original_path)
