About SMUTHI
===============

SMUTHI stands for 'scattering by multiple particles in thin-film systems'.
The software allows you to solve light scattering problems involving
one ore multiple particles near or inside a system of planar layer interfaces.
It is based on the T-matrix method for the single particle scattering,
and on the scattering-matrix method for the propagation through the layered medium.

.. image:: images/drawing.png
   :scale: 40%
   :align: center

The software solves Maxwell's equations (3D wave optics) in frequency domain (one wavelength per simulation).
An arbitrary number of spheres, spheroids and finite cylinders inside an arbitrary system of plane parallel layers can
be modelled. For spheres, the T-matrix is given by the Mie-coefficients. For spheroids and finite cylinders, SMUTHI
calls the
`NFM-DS <https://scattport.org/index.php/programs-menu/t-matrix-codes-menu/239-nfm-ds>`_,
to compute the single particle T-matrix. This is a Fortran software package written by A. Doicu, T. Wriedt and Y. Eremin, based on the "Null-field method with
discrete sources".

.. image:: images/norm_E.png
   :scale: 52 %

.. image:: images/E_y.gif
   :scale: 52 %

With Smuthi, you can compute the 3D electric near field along a cut plane and save it in the form of ascii data files,
png images or animations. The dashed circles around the particles are a reminder that inside the circumscribing sphere
of the particles, the computed near fields cannot be trusted.

.. image:: images/bottom_dcs.png
   :scale: 52 %

.. image:: images/bottom_polar_dcs.png
   :scale: 52 %

In addition, the far field power flux can be evaluated. For plane wave incidence, it is normalized by the incoming
wave's intensity to yield the
`differential cross section <https://en.wikipedia.org/wiki/Cross_section_(physics)#Differential_cross_section>`_.
The above images show the 2D differential cross section in the
bottom layer as a polar plot (left) and its azimuthal integral as a function of the polar angle only (right)

.. math:: \mathrm{DCS}_\mathrm{polar}(\beta) = \int \mathrm{d} \alpha \, \sin\beta \, \mathrm{DCS}(\beta, \alpha)

where :math:`(\alpha,\beta)` are the azimuthal and polar angle, respectively.

The sharp feature around 40° in the shown example relates to total internal reflection at the interface between media 2
and 3.

Further, Smuthi also returns the extinction cross sections for the reflected and the transmitted wave. For the
scattering of a plane wave by particles in a homogeneous medium, the extinction cross section is usually defined as the
sum of total scattering and absorption cross section.

In Smuthi, we instead use what is usually referred to as the
`optical theorem <https://en.wikipedia.org/wiki/Optical_theorem>`_ to define extinction. That means, the extinction
cross section for reflection (transmission) refers to the destructive interference of the scattered signal with the
specular reflection (transmission) of the initial wave. It thereby includes absorption in the particles, scattering,
and a modified absorption by the layer system, e.g. through incoupling into waveguide modes. If the particles lead to,
say, a higher reflection than the bare layer system without particles, the extinction can also be negative.