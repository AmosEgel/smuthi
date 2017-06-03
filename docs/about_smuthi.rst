About SMUTHI
===============

SMUTHI stands for 'scattering by multiple particles in thin-film systems'.
The software allows you to solve light scattering problems involving
one ore multiple particles near or inside a system of planar layer interfaces.
It is based on the T-matrix method for the single particle scattering,
and on the scattering-matrix method for the propagation through the layered medium.

.. image:: images/drawing.png
   :scale: 40%
.. image:: images/norm_E.png
   :scale: 60 %
The software solves Maxwell's equations (3D wave optics) in frequency domain (one wavelength per simulation).
An arbitrary number of spheres, spheroids and finite cylinders inside an arbitrary system of plane parallel layers can
be modelled. For spheres, the T-matrix is given by the Mie-coefficients. For spheroids and finite cylinders, SMUTHI
calls the
`NFM-DS <https://scattport.org/index.php/programs-menu/t-matrix-codes-menu/239-nfm-ds>`_,
to compute the single particle T-matrix. This is a Fortran software package written by A. Doicu, T. Wriedt and Y. Eremin, based on the "Null-field method with
discrete sources".

SMUTHI returns:
 - total and differential scattering cross sections as well as extinction cross sections
 - electric near fields (plots or animations)
