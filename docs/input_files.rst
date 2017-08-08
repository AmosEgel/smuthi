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

In addition, specify :code:`l_max` and :code:`m_max`, which refer to the maximal multipole degree and order used for the
spherical wave expansion of that particle's scattered field. These parameters should be chosen with reference to the
desired accuracy and to the particle size parameter and refractive index contrast, see for example
https://arxiv.org/ftp/arxiv/papers/1202/1202.5904.pdf
A larger value leads to higher accuracy, but also to longer computation time. :code:`l_max` is a positive integer and
:code:`m_max` is a non-negative integer and not greater than :code:`l_max`.

In the case of non-spherical particles, you can also specify :code:`use discrete sources` (default is :code:`True`),
:code:`nint` (default is :code:`200`) and :code:`nrank: 8` (default is :code:`l_max + 2`). These parameters specify the
calculation of the T-matrix using the NFM-DS module. For further information about the meaning of these parameters, see
the `NFM-DS documentation <https://scattport.org/images/scattering-code/NFM-DS_program-description.pdf>`_.

The parameters for the scattering particles can be listed directly in the input file, in the following format::


  scattering particles:
  - shape: sphere
    radius: 100
    refractive index: 2.4
    extinction coefficient: 0.05
    position: [0, 100, 150]
    l_max: 3
    m_max: 3
  - shape: finite cylinder
    cylinder radius: 120
    cylinder height: 150
    refractive index: 2.7
    extinction coefficient: 0
    position: [350, -100, 250]
    euler angles: [0, 0, 0]
    l_max: 4
    m_max: 4
    use discrete sources: true
    nint: 200
    nrank: 8
  - shape: spheroid
    semi axis c: 80
    semi axis a: 140
    refractive index: 2.5
    extinction coefficient: 0.05
    position: [-350, 50, 350]
    euler angles: [0, 0, 0]
    l_max: 3
    m_max: 3
    use discrete sources: true
    nint: 200
    nrank: 8


Alternatively, the scattering particles can be specified in a separate file, which needs to be located in the SMUTHI
project folder.
This is more convenient for large particle numbers. 
In that case, specify the filename of the particles parameters file, for example::

   scattering particles: particle_specs.dat

The format of the particle specifications file is described below, see `The particle specifications file`_.

Initial field
---------------

Currently, only plane waves are implemented as the initial excitation. 

Specify the initial field in the following format::

  initial field:
    type: plane wave
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

Specify the contour of the sommerfeld integral in the complex :code:`neff` plane where :code:`neff = k_parallel / omega` refers to the effective refractive index of the partial wave. The contour is parameterized by its waypoints::

   neff waypoints: [0, 0.5, 0.8-0.1j, 2-0.1j, 2.5, 4]

as well as its discretization scale::

   neff discretization: 1e-3

The :code:`neff waypoints` define a piecewise linear trajectory in the complex plane. This trajectory should start at
:code:`0` and end at a suitable real truncation parameter (somewhere above the highest layer refractive index).
A simple contour would be for example :code:`neff waypoints: [0, 4]`. However
The trajectory can be deflected into the lower complex half plaen such that it does not come close to waveguide mode
resonances of the layer system.


Post procesing
-----------------

Define here, what output you want to generate. Currently, the following tasks can be defined for the post processing
phase:

  - evaluation of scattering and extinction cross sections
  - evaluation of the electrical near field

Write for example::

   post processing:
   - task: evaluate cross sections
     show plots: false
     save plots: true
     save data: false
   - task: evaluate near field
     show plots: false
     save plots: true
     save animations: true
     save data: false
     quantities to plot: [E_y, norm(E), E_scat_y, norm(E_scat), E_init_y, norm(E_init)]
     xmin: -800
     xmax: 800
     zmin: -400
     zmax: 900
     spatial resolution: 50
     interpolation spatial resolution: 5
     maximal field strength: 1.2

The :code:`show plots`, :code:`save plots` and :code:`save data` flags deterimine, if the respective output
is plotted, if the plots are saved and if the raw data is exported to ascii files.

In the :code:`evaluate near field` task, the :code:`save animations` flags deterimines, if the near field figures are
exported as gif animations.

The :code:`quantities to plot` are a list of strings that can be:
:code:`E_x`, :code:`E_y`, :code:`E_z` or :code:`norm(E)` for the x-, y- and z-component or the norm of the total
electric field,
:code:`E_scat_x`, :code:`E_scat_y`, :code:`E_scat_z` or :code:`norm(E_scat)` for the x-, y- and z-component or the norm
of the scattered electric field,
or :code:`E_init_x`, :code:`E_init_y`, :code:`E_init_z` or :code:`norm(E_init)` for the x-, y- and z-component or the norm
of the initial electric field.

To specify the plane where the near field is computed, provide :code:`xmin`, :code:`xmax`, :code:`ymin`, :code:`ymax`,
:code:`zmin` and :code:`zmax`. If any of these is not given, it is assumed to be 0.
For exactly one of the coordinates x, y or z the min and max value should be identical, e.g. :code:`ymin` =
:code:`ymax` as in the above example. In that case, the field is plotted in the xz-plane.

:code:`spatial resolution` determines, how fine the grid of points is, where the near field is computed.
As :code:`xmin` etc., this parameter is specified in length units. If :code:`interpolation spatial resolution` is
specified, the near field will be interpolated to that finer value to allow for smoother looking field plots without the
long computing time of a fine grained actual field evaluation.

With :code:`maximal field strength`, you can set the color scale of the field plots to a fixed maximum.


Further settings for the generation of output data
---------------------------------------------------

The path to the output folder can be specified as::

   output folder: smuthi_output

This folder will be created and in it a subfolder with a timestamp that contains all file output of the simulation.

Finally, if::

   save simulation: true

is specified, the simulation object will be saved as a binary data file from which it can be reimported at a later time.



The particle specifications file
==================================

The file containing the particle specifications needs to be written in the following format::


   # spheres
   # x, y, z, radius, refractive index, exctinction coefficient, l_max, m_max
   0        100     150     100     2.4     0.05    3       3
   ...      ...     ...     ...     ...     ...     ...     ...

   # cylinders
   # x, y, z, cylinder radius, cylinder height, refractive index, exctinction coefficient, l_max, m_max
   250      -100    250	    120     150     2.7     0       4       4
   ...      ...     ...     ...     ...     ...     ...     ...     ...

   # spheroids
   # x, y, z, semi-axis c, semi-axis a, refractive index, exctinction coefficient, l_max, m_max
   -250     0       350     80      140     2.5     0.05    3       3
   ...      ...     ...     ...     ...     ...     ...     ...     ...

An examplary particle specifiacations can be downloaded from
:download:`here <../smuthi/data/example_particle_specs.dat>`.

Back to :doc:`main page <index>`
