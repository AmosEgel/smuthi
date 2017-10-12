SMUTHI
=======================
SMUTHI stands for 'scattering by multiple particles in thin-film systems'. The software allows to simulate light
scattering by multiple particles near (or between) planar interfaces. It is based on the T-matrix method for the single
particle scattering, and on the scattering-matrix method for the propagation through the layered medium.

Target group: Scientists and engineers in the field of optics and optoelectronics.

License: SMUTHI is provided under the MIT license.

Author: Amos Egel. Mail to amos.egel@kit.edu for support or to report a problem.

For a guide how to install and use the software, see the `documentation <http://smuthi.readthedocs.io>`_.

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

 - faster evaluation of particle coupling
 - faster (and less memory intense) evaluation of near field
