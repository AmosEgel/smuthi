# -*- coding: utf-8 -*-

from setuptools import setup

readme = open('README.rst', 'r')
README_TEXT = readme.read()
readme.close()


setup(
    name="SMUTHI",
    version="0.1.1",
    author="Amos Egel",
    author_email="amos.egel@kit.edu",
    url='https://gitlab.com/AmosEgel/smuthi',
    description="Light scattering by multiple particles in thin-film systems",
    long_description=README_TEXT,
    packages=['smuthi'],
    install_requires=['numpy', 'scipy', 'sympy', 'matplotlib', 'pyyaml', 'argparse'],
    entry_points={'console_scripts': ['smuthi = smuthi.__main__:main']},
    license='MIT'
)
