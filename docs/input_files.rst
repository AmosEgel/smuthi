============
Input files
============

Parameters specified in the input file
=======================================

In the following, the parameters which can be specified in the input file are listed:

Length unit
------------
Declare here the unit in which you want to specify all lengths. 
It has no influence on the calculations and can be chosen arbitrarily. 
This field is mainly there to remind the user that all lengths have to be specified in consistent units. 
In addition, it is used for the axis annotation of output plots::

   length unit: nm

Vacuum wavelength
------------------

The vacuum wavelength :math:`\lambda` of the electromagnetic field, in the specified length unit::

   vacuum wavelength: 550

Layer system
---------------

Define the background geometry of the layered medium. 
A layer system consists of :math:`N` layers, counted from bottom to top. 
Each layer is characterized by its thickness as well as its (real) refractive index :math:`n` and extinction coefficient :math:`k`
(the latter is equivalent to the imaginary part of the complex refractive index :math:`\tilde{n}=n+jk`). 
Provide the thickness information in the form of :math:`[d_0, d_1, ..., d_N]`, where :math:`d_i` is the thickness of the :math:`i`-th layer. 
As the outermost layers are infinitely thick, specify them with a thickness of :math:`0`. 
Analogously, provide the refractive indices and extinction coefficients in the form of :math:`[n_0, ..., n_N]` and :math:`[k_0, ..., k_N]`.

For example, the following entry::

   layer system:
   - thicknesses: [0, 500, 0]
     refractive indices: [1.5, 2.1, 1]
     extinction coefficients: [0, 0.01, 0]

would specify a single film of thickness :math:`500`, consisting of a material with complex refractive index :math:`n_1=2.1+0.01j`, located on top of a substrate with refractive index :math:`n_0=1.5`, and below air/vacuum (refractive index :math:`n_2=1`).

Scattering particles
---------------------

The ensemble of scattering particles inside the layered medium.

For spherical particles, specify
:code:`shape: sphere`, the radius, refractive index, extinction coefficient 
and the :code:`[x, y, z]` coordinates of the particle position.

For spheroids, specify
:code:`shape: spheroid`, the half axes along (`half axis c`) and transverse (`half axis a`) to the axis of revolution,
refractive index, extinction coefficient and the :code:`[x, y, z]` coordinates of the particle position, as well as the
Euler angles defining the rotation of the axis of revolution relative to the `z` axis (currently rotations other than
`[0, 0, 0]` are not implemented).

For finite cylinders, specify
:code:`shape: finite cylinder`, the cylinder height, cylinder radius, refractive index, extinction coefficient and the
:code:`[x, y, z]` coordinates of the particle position, as well as the
Euler angles defining the rotation of the axis of revolution relative to the `z` axis (currently rotations other than
`[0, 0, 0]` are not implemented).


The coordinate system is such that the interface between the first two layers defines the plane :math:`z=0`.

The parameters can be listed directly in the input file, in the following format::

   scattering particles:
   - shape: sphere
     radius: 100
     refractive index: 2.4
     extinction coefficient: 0.05
     position: [0, 100, 150]
   - shape: finite cylinder
     cylinder radius: 120
     cylinder height: 150
     refractive index: 2.7
     extinction coefficient: 0
     position: [250, -100, 250]
     euler angles: [0, 0, 0]
   - shape: spheroid
     semi axis c: 80
     semi axis a: 140
     refractive index: 2.5
     extinction coefficient: 0.05
     position: [-250, 0, 350]
     euler angles: [0, 0, 0]

Alternatively, the scattering particles can be specified in a separate file, which needs to be located in the SMUTHI project folder. 
This is more convenient for large particle numbers. 
In that case, specify the filename of the particles parameters file, for example::

   scattering particles: particle_specs.dat

The format of the particle specifications file is described below, see `The particle specifications file`_.

Initial field
---------------

Currently, only plane waves are implemented as the initial excitation. 

Specify the initial field in the following format::

   initial field:
   - type: plane wave
     angle units: degree
     polar angle: 0
     azimuthal angle: 0
     polarization: TE
     amplitude: 1
     reference point: [0, 0, 0]

Angle units can be 'degree' (otherwise, radians are used). For polarization, select either :code:`TE` or :code:`TM`. 

The electric field of the plane wave in the layer from which it comes then reads

.. math:: \mathbf{E_\mathrm{init}}(\mathbf{r}) = A \exp(\mathrm{j} \mathbf{k}\cdot(\mathbf{r}-\mathbf{r_0})) \hat{\mathbf{e}}_j,

where :math:`A` is the amplitude, :math:`\mathrm{j}` is the imaginary unit,

.. math:: \mathbf{k}=\frac{2 \pi n_\mathrm{init}}{\lambda}  \left( \begin{array}{c} \sin(\beta)\cos(\alpha)\\ \sin(\beta)\sin(\alpha) \\ \cos(\beta) \end{array} \right)

is the wave vector in the layer from which the plane wave comes,
:math:`n_\mathrm{init}` is the refractive index in that layer (must be real), :math:`(\beta,\alpha)` are the polar and azimuthal angle of the plane wave,
:math:`\mathbf{r_0}` is the reference point and 
:math:`\hat{\mathbf{e}}_j` is the unit vector pointing into the :math:`\alpha`-direction for :code:`TE` polarization 
and into the  in the :math:`\beta`-direction for :code:`TM` polarization.

If the polar angle is in the range :math:`0\leq\beta\lt 90^\circ`, the k-vector has a positive :math:`z`-component and consequently, the plane wave is incident from the bottom side. 
If the polar angle is in the range :math:`90^\circ\lt\beta\leq 180^\circ`, then the plane wave is incident from the top. 


Numerical parameters
----------------------

Specify the multipole truncation degree :code:`lmax` and order :code:`mmax`, for example::

   lmax: 3

   mmax: 3

:code:`lmax` and :code:`mmax` should be chosen with reference to the desired accuracy and to the particle size parameter and refractive index contrast, see for example https://arxiv.org/ftp/arxiv/papers/1202/1202.5904.pdf
A larger value leads to higher accuracy, but also to longer computation time. :code:`lmax` is a positive integer and :code:`mmax` is a non-negative integer and not greater than :code:`lmax`.

Further, specify the contour of the sommerfeld integral in the complex :code:`neff` plane where :code:`neff = k_parallel / omega` refers to the effective refractive index of the partial wave. The contour is parameterized by its waypoints::

   neff waypoints: [0, 0.5, 0.8-0.1j, 2-0.1j, 2.5, 4]

as well as its discretization scale::

   neff discretization: 1e-3

The :code:`neff waypoints` define a piecewise linear trajectory in the complex plane. This trajectory should start at
:code:`0` and end at a suitable real truncation parameter (somewhere above the highest layer refractive index).
A simple contour would be for example :code:`neff waypoints: [0, 4]`. However
The trajectory can be deflected into the lower complex half plaen such that it does not come close to waveguide mode
resonances of the layer system.

T-matrix method for non-spherical particles
-------------------------------------------
Spheroids can currently be modelled using the NFM-DS method. Specify the parameters for the algorithm like this::

   tmatrix method:
   - algorithm: nfm-ds
     use discrete sources: true
     nint: 200
     nrank: 8

The :code:`use discrete sources` flag determines, if during the NFM-DS method, the discrete sources functionality is
activated. Generally, it leads to a better accuracy for particle shapes deviating strongly from that of a sphere.
:code:`nint` is the truncation multipole degree used inside the NFM-DS algorithm, and is by default set to
:code:`lmax + 2`. :code:`nrank` is a parameter that specifies how fine the numerical integrations in the NFM-DS are
discretized. See the
`NFM-DS documentation <https://scattport.org/images/scattering-code/NFM-DS_program-description.pdf>`_ for further
details information.



Post procesing
-----------------

Define here, what output you want to generate. Currently only the evaluation of scattering and extinction cross sections is implemented. Write::

   post processing:
   - task: evaluate cross sections
     show plots: true

If :code:`show plots` is not set to :code:`false` (default), the differential scattering cross section is plotted.

The particle specifications file
==================================

The file containing the particle specifications needs to be written in the following format::

   # spheres
   # x, y, z, radius, refractive index, exctinction coefficient
   0	    100		150		100		2.4		0.05
   ...      ...     ...     ...     ...     ...

   # cylinders
   # x, y, z, cylinder radius, cylinder height, refractive index, exctinction coefficient
   250      -100    250	    120     150     2.7     0
   ...      ...     ...     ...     ...     ...     ...

   # spheroids
   # x, y, z, semi-axis c, semi-axis a, refractive index, exctinction coefficient
   -250	    0       350	    80      140     2.5     0.05
   ...      ...     ...     ...     ...     ...     ...

An examplary particle specifiacations can be downloaded from
:download:`here <../smuthi/data/example_particle_specs.dat>`.

Back to :doc:`main page <index>`
