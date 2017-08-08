# -*- coding: utf-8 -*-
import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="SMUTHI",
    version="0.4.0",
    author="Amos Egel",
    author_email="amos.egel@kit.edu",
    url='https://gitlab.com/AmosEgel/smuthi',
    description="Light scattering by multiple particles in thin-film systems",
    long_description=read('README.rst'),
    packages=['smuthi', 'smuthi.nfmds'],
    package_data={'smuthi.nfmds': ['NFM-DS/*.txt', 'NFM-DS/TMATSOURCES/*', 'NFM-DS/TMATFILES/*',
                                   'NFM-DS/INPUTFILES/*.dat', 'NFM-DS/OUTPUTFILES/*'],
                  'smuthi': ['data/*']},
    install_requires=['numpy', 'scipy', 'sympy', 'matplotlib', 'pyyaml', 'argparse', 'imageio'],
    entry_points={'console_scripts': ['smuthi = smuthi.__main__:main']},
    license='MIT',
)
