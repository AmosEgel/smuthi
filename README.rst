.. image:: https://gitlab.com/AmosEgel/smuthi/badges/master/pipeline.svg

.. image:: https://readthedocs.org/projects/smuthi/badge/?version=latest
   :target: https://smuthi.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
   :align: right



|

.. image:: https://gitlab.com/AmosEgel/smuthi/raw/master/docs/images/logo_cropped.png
   :align: center

   

SMUTHI stands for 'scattering by multiple particles in thin-film systems'. The software allows to simulate light
scattering by multiple particles near (or between) planar interfaces. It is based on the T-matrix method for the single
particle scattering, and on the scattering-matrix method for the propagation through the layered medium.

Target group: Scientists and engineers in the field of optics and optoelectronics.

License: SMUTHI is provided under the MIT license.

Author: Amos Egel (amos.egel@gmail.com).

The following persons have contributed to the project: Dominik Theobald (functions for the simulation of rotated particles, 
plane wave based particle coupling for non-spherical particles with overlapping circumscribing spheres), Lorenzo Pattelli (logo),
Konstantin Ladutenko (MPI support, numerous additions and improvements for a smooth user experience, correction of bugs)

We thank Adrian Doicu, Thomas Wriedt and Yuri Eremin for allowing us to use their NFM-DS Fortran code, 
Ilia Rasskazov as well as Fabio Mangini for bug reports and useful comments 
and Håkan T Johansson for making his pywigjxpf software availible through PyPi and also under Windows.

For a guide how to install and use the software, see the `documentation <http://smuthi.readthedocs.io>`_.

If you are using Smuthi, please `subscribe to the Smuthi mailing list <https://groups.google.com/forum/#!forum/smuthi/join>`_.
The list is also a good place to ask for support from the developers or from experienced users.

To report a bug, you can also open an issue in Gitlab.

Contributions are highly welcome! Please refer to the `contribution guidelines <https://gitlab.com/AmosEgel/smuthi/blob/master/CONTRIBUTING.rst>`_.

What's new in version 1.0
--------------------------
A major bug that significantly slowed down Smuthi under Windows was fixed. 
The module structure has undergone a major review (unfortunately, backwards compatibility cannot be granted and you might need to adapt some import statements in your scripts when updating to version 1.0). 
For spheres, the calculation of internal fields (i.e., inside the particle) was implemented.
A module for automatic selection of numerical parameters has been added (still in beta).
For non-spherical particles, Smuthi now requires the GNU Fortran compiler also under Windows (MinGW). The use of the precompiled executable is deprecated.
Pywigxjpf is now also available for Windows - however, the sympy fallback solution for the calculation of the Wigner3j symbols is still provided.
Convenience functions for the definition of reasonable Sommerfeld contours have been added and can be managed through the simulation class (call to "set_default_k_parallel" no more necessary).
Plenty of smaller changes and bug fixes. 
Advanced logging. 


What's new in version 0.9
-------------------------
MPI support for the parallel execution of many simulations, acceleration with Numba JIT, faster evaluation of Wigner3j symbols through pywigxjpf

What's new in version 0.8
-------------------------
Support for rotated particles, GPU support for the calculation of the near field.  

What's new in version 0.7
--------------------------
Iterative solver (GMRES), lookup tables and GPU support were added for fast simulations including large particle
numbers.

What's new in version 0.6
--------------------------
Dipole sources are supported as initial field.

What's new in version 0.5
--------------------------
Gaussian beams (more precisely: beams with transverse Gaussian footprint) are supported as initial field.

What's new in version 0.4
--------------------------
The data structure has been updated to a more consequent object oriented approach, including a PlaneWaveExpansion class
and a SphericalWaveExpansion class. Smuthi's API is now also `documented <http://smuthi.readthedocs.io>`_.

What's new in version 0.3
--------------------------
The software now allows to compute the electric near field. The fields can be plotted as png figure files and as gif
animations. All generated output can be stored as figure files or as text files. The simulation object can be exported
as binary file.

What's new in version 0.2.2
---------------------------
Finite cylinders were added.

What's new in version 0.2
--------------------------
In addition to spherical particles, spheroids can now be selected as scattering particles, too.
Spheroids are ellipsoidal particles with one axis of rotational symmetry (which is currently fixed
to be the direction perpendicular to the layer interfaces).
