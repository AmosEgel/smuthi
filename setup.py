# -*- coding: utf-8 -*-

import setuptools
from setuptools.command.install import install
import sys
import subprocess
from numpy.distutils.core import Extension
from numpy.distutils.core import setup

with open('README.rst', 'r') as readme:
    README_TEXT = readme.read()

setup(
    name="SMUTHI",
    version="0.2.a",
    author="Amos Egel",
    author_email="amos.egel@kit.edu",
    url='https://gitlab.com/AmosEgel/smuthi',
    description="Light scattering by multiple particles in thin-film systems",
    long_description=README_TEXT,
    packages=['smuthi', 'smuthi.nfmds'],
    package_data={'smuthi.nfmds': ['data/*'],
                  'smuthi': ['data/*']},
    install_requires=['numpy', 'scipy', 'sympy', 'matplotlib', 'pyyaml', 'argparse'],
    entry_points={'console_scripts': ['smuthi = smuthi.__main__:main']},
    license='MIT',
    ext_modules=[Extension('smuthi.nfmds.taxsym',
                           ['smuthi/nfmds/fortran_sources/TAXSYM.f90'])],
)
