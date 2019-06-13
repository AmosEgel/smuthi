.. image:: docs/images/logo_cropped.png
   :align: center

SMUTHI stands for 'scattering by multiple particles in thin-film systems'. The software allows to simulate light
scattering by multiple particles near (or between) planar interfaces. It is based on the T-matrix method for the single
particle scattering, and on the scattering-matrix method for the propagation through the layered medium.

Target group: Scientists and engineers in the field of optics and optoelectronics.

License: SMUTHI is provided under the MIT license.

Author: Amos Egel (amos.egel@gmail.com).

The following persons have contributed to the project:

 - Dominik Theobald (functions for the simulation of rotated particles, plane wave based particle coupling for 
   non-spherical particles with overlapping circumscribing spheres) 
 - Lorenzo Pattelli (logo)
 - Konstantin Ladutenko (numerous additions and improvements for a smooth user experience, correction of bugs)

We thank Adrian Doicu, Thomas Wriedt and Yuri Eremin for allowing us to use their NFM-DS Fortran code, 
and Ilia Rasskazov for bug reports and useful comments.

For a guide how to install and use the software, see the `documentation <http://smuthi.readthedocs.io>`_.

If you are using Smuthi, please subscribe to the `Smuthi mailing list <https://groups.google.com/forum/#!forum/smuthi>`_.
The list is also a good place to ask for support from the developers or from experienced users.

To report a bug, you can also open an issue in Gitlab.

Contributions are highly welcome! Please refer to the `contribution guidelines <https://gitlab.com/AmosEgel/smuthi/blob/master/CONTRIBUTING.rst>`_.


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
--------------------------
Finite cylinders were added.

What's new in version 0.2
--------------------------
In addition to spherical particles, spheroids can now be selected as scattering particles, too.
Spheroids are ellipsoidal particles with one axis of rotational symmetry (which is currently fixed
to be the direction perpendicular to the layer interfaces).


Planned updates for future versions
------------------------------------
Things to be implemented next:

 - plane wave based coupling for close non-spherical particles (by Dominik Theobald, in progress)
 - faster runtime when running on CPU only (by Konstantin Ladutenko)


