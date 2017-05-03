.. SMUTHI documentation master file, created by
   sphinx-quickstart on Wed May  3 19:50:09 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   
Welcome to SMUTHI's documentation!
==================================
SMUTHI means 'Scattering by MUltiple particles in THIn-film systems'. The software allows you to simulate light scattering
by multiple particles near or iside a system of planar layer interfaces. It is based on the T-matrix method for the single particle
scattering, and on the scattering-matrix method for the propagation through the layered medium.
  
Installation
=============
SMUTHI is implemented in Python 3.5. To run the program, you first need to make sure that Python 3 is installed on your computer.
If this is not the case, see :doc:`here<how_install_python>`.

Installing SMUTHI
-------------------
Using pip, you can install SMUTHI simply by::

   pip install smuthi

Alternatively, you can download the SMUTHI project folder manually from `here <https://gitlab.com/AmosEgel/smuthi>`_. 
Open a command prompt and change directory to the SMUTHI project folder. Then, enter (Windows)::

   python setup.py install

or (Ubuntu)::

   python3 setup.py install


Running simulations
======================
SMUTHI is executed from the command line together with one argument, 
specifying the input file that contains all parameters of the configuration to be simulated, 
see `The input file`_.

Open a command window (shell or Win Python Command Prompt) and type::

   smuthi path/to/input.dat

If :code:`smuthi` is called without an argument, it tries to open :code:`input.dat` as default in the local folder.
   
The input file
===============
The input file uses the `YAML <http://yaml.org/>`_ format. An example file :code:`input.dat` is contained in the SMUTHI project folder and can be downloaded :download:`here <../input.dat>`. 
You can modify the entries to adapt the file to your use case.

In the following, the parameters which can be specified in the input file are listed:

Length unit
-----------------
Declare here the unit in which you want to specify all lengths. It has no influence on the calculations and can be chosen arbitrarily. This field is mainly there to remind the user that all lengths have to be specified in consistent units. In addition, it is used for the axis annotation of output plots::

   length unit: nm

Vacuum wavelength
------------------
The vacuum wavelength of the electromagnetic field, in the specified length unit::

   vacuum wavelength: 550

Layer system
--------------
Define the background geometry of the layered medium. 
A layer system consists of :code:`N` layers, counted from bottom to top. 
Each layer is characterized by its thickness as well as its (real) refractive index and extinction coefficient 
(the latter is equivalent to the imaginary part of the complex refractive index). 
Provide the thickness information in the form of :code:`[d0, d1, d2, ..., dN]`, where :code:`di` is the thickness of layer :code:`i`. 
As the outermost layers are infinitely thick, specify them with a thickness of :code:`0`. 
Analogously, provide the refractive indices and extinction coefficients in the form of :code:`[n0, ..., nN]` and :code:`[k0, ..., kN]`.

For example, the following entry::

   layer system:
   - thicknesses: [0, 500, 0]
     refractive indices: [1.5, 2.1, 1]
     extinction coefficients: [0, 0.01, 0]

would specify a single film of thickness :code:`500`, consisting of a material with complex refractive index :code:`2.1+0.01j`, located on top of a substrate with refractive index :code:`1.5`, and below air/vacuum (refractive index :code:`1`).

Scattering particles
---------------------
The ensemble of scattering particles inside the layered medium. For spherical particles, specify 
:code:`shape: sphere`, the radius, refractive index, extinction coefficient 
and the :code:`[x, y, z]` coordinates of the particle position. 
**The coordinate system is such that the interface between the first two layers defines the plane** :code:`z=0`

The parameters can be listed directly in the input file, in the following format::

   scattering particles:
   - shape: sphere
     radius: 100
     refractive index: 2.4
     extinction coefficient: 0
     position: [220, 110, 250]
   - shape: sphere
     radius: 200
     refractive index: 2.1
     extinction coefficient: 0.01
     position: [-300, -200, 750]

Alternatively, the scattering particles can be specified in a separate file, which needs to be located in the SMUTHI project folder. 
This is more convenient for large particle numbers. 
In that case, specify the filename of the particles parameters file, for example::

   scattering particles: particle_specs.dat

The format of the particle specifications file is described below, see `The particle specifications file`_.

Initial field
--------------
Currently, only plane waves are implemented as the initial excitation. Specify the initial field in the following format::

   initial field:
   - type: plane wave
     angle units: degree
     polar angle: 0
     azimuthal angle: 0
     polarization: TE
     amplitude: 1
     reference point: [0, 0, 0]

Polar and azimuthal angle refer to the corresponding spherical coordinates of the plane wave's wave vector. Angle units can be 'degree' (otherwise, radians are used). If the polar is between :code:`0` and :code:`90` degree, the k-vector has a positive z-component and consequently, the plane wave is incident from the bottom side. If the polar angle is between :code:`90` and :code:`180` degree, then the plane wave is incident from the top. For polarization, select either :code:`TE` or :code:`TM`. The reference point specifies the location where the incident wave would have zero phase, that is, the electric field of the incident wave is proportional to :code:`E(r) = A * exp(j k.(r-r0))` where :code:`A` is the amplitude and :code:`r0` is the reference point.

Numerical parameters
-----------------------
Specify the multipole truncation degree :code:`lmax` and order :code:`mmax`, for example::

   lmax: 3

   mmax: 3

:code:`lmax` and :code:`mmax` should be chosen with reference to the desired accuracy and to the particle size parameter and refractive index contrast, see for example https://arxiv.org/ftp/arxiv/papers/1202/1202.5904.pdf
A larger value leads to higher accuracy, but also to longer computation time. :code:`lmax` is a positive integer and :code:`mmax` is a non-negative integer and not greater than :code:`lmax`.

Further, specify the contour of the sommerfeld integral in the complex :code:`neff` plane where :code:`neff = k_parallel / omega` refers to the effective refractive index of the partial wave. The contour is parameterized by its waypoints::

   neff waypoints: [0, 0.5, 0.8-0.1j, 2-0.1j, 2.5, 4]

as well as its discretization scale::

   neff discretization: 1e-3

The :code:`neff waypoints` define a piecewise linear trajectory in the complex plane. This trajectory should start at :code:`0` and end at a suitable real truncation parameter (somewhere above the highest layer refractive index). 
A simple contour would be for example :code:`neff waypoints: [0, 4]`. However
The trajectory can be deflected into the lower complex half plaen such that it does not come close to waveguide mode resonances of the layer system.

Post procesing
-----------------
Define here, what output you want to generate. Currently only the evaluation of scattering and extinction cross sections is implemented. Write::

   post processing:
   - task: evaluate cross sections
     show plots: true

If :code:`show plots` is not set to :code:`false` (default), the differential scattering cross section is plotted.

The particle specifications file
===================================
The file containing the particle specifications needs to be written in the following format::

   # spheres
   # x         y           z           radius      ref. idx.   exct. coeff.
   220         110         250         100         2.4         0
   -300        -200        750         200         2.1         0.01
   ...         ...         ...         ...         ...         ...
   
An examplary particle specifiacations file with the name :code:`particle_specs.dat` is provided in the SMUTHI project folder and can be downloaded  :download:`here <../particle_specs.dat>`.



