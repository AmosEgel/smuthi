# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name="SMUTHI",
    version="0.1",
    author="Amos Egel",
    author_email="amos.egel@kit.edu",
    description="SMUTHI stands for 'Scattering by MUltiple particles in THIn-film systems'.",
    long_description=("SMUTHI is based on the T-matrix method for the single particle scattering, and on the "
                      "scattering-matrix method for the propagation through the layered medium."),
    packages=find_packages(),
    entry_points={'console_scripts': ['smuthi = smuthi.__main__:main']},
)
