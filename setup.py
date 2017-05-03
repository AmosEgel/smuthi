# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    name="SMUTHI",
    version="0.1",
    author="Amos Egel",
    author_email="amos.egel@kit.edu",
    url='https://gitlab.com/AmosEgel/smuthi',
    description="Light scattering by multiple particles in thin-film systems",
    long_description=("SMUTHI allows to simulate light scattering by multiple particles near (or between) planar "
                      "interfaces. It is based on the T-matrix method for the single particle scattering, and on the "
                      "scattering-matrix method for the propagation through the layered medium."),
    packages=['smuthi'],
    install_requires=['numpy', 'scipy', 'sympy', 'matplotlib', 'pyyaml', 'argparse'],
    entry_points={'console_scripts': ['smuthi = smuthi.__main__:main']},
    license='MIT'
)
