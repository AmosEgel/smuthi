About Smuthi
===============

Smuthi stands for 'scattering by multiple particles in thin-film systems'.
It is a Python software that allows to solve light scattering problems involving
one ore multiple particles near or inside a system of planar layer interfaces.

.. image:: images/drawing.png
   :scale: 40%
   :align: center

It solves the Maxwell equations (3D wave optics) in frequency domain (one wavelength per simulation).


Simulation method
------------------

Smuthi is based on the T-matrix method for the single particle scattering and on the scattering-matrix method for the propagation through the layered medium. This combination of methods is described in the following papers:

  - `Egel, Amos, and Uli Lemmer. JQSRT 148 (2014): 165-176. <https://www.sciencedirect.com/science/article/pii/S0022407314002829?via%3Dihub>`_
  - `Egel, Amos, Siegfried W. Kettlitz, and Uli Lemmer. JOSA A 33.4 (2016): 698-706. <https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-33-4-698>`_
  - Egel, Amos: PhD thesis (in preparation)

For spheres, the T-matrix of the individual particles is given by the Mie-coefficients. 
For spheroids and finite cylinders, Smuthi calls the 
`NFM-DS <https://scattport.org/index.php/programs-menu/t-matrix-codes-menu/239-nfm-ds>`_
to compute the single particle T-matrix. This is a Fortran software package written by 
A. Doicu, T. Wriedt and Y. Eremin, based on the "Null-field method with discrete sources", see

  - `Doicu, Adrian, Thomas Wriedt, and Yuri A. Eremin. Light scattering by systems of particles: null-field method with discrete sources: theory and programs. Vol. 124. Springer, 2006. <http://www.springer.com/us/book/9783540336969>`_


Performance critical parts of the software are implemented in CUDA. When dealing with a large number of particles, Smuthi can benefit from a substantial acceleration if a suitable (NVIDIA) GPU is available.

For CPU-only execution, other acceleration concepts (including MPI parallelization, Numba JIT compilation) are currently tested. 


Range of applications
----------------------

Smuthi can be applied to any scattering problem in frequency domain involving

  - a system of plane parallel layer interfaces separating an arbitrary number of metallic or dielectric layers.

  - an arbitrary number of wavelength-scale scattering particles (currently available: spheres, spheroids, finite cylinders). The particles can be metallic or dielectric and rotated to an arbitrary orientation.

  - an initial field in form of a plane wave, a beam (currently available: beam with Gaussian xy-profile) or a collection of dipole sources

Thus, the range of applications spans from scattering by a single particle on a substrate to scattering by several thousand particles inside a planarly layered medium. For a number of examplary simulations, see the :doc:`examples<gallery>` section.


Simulation output
------------------

Smuthi can compute

  - the 3D electric field, for example along a cut plane and save it in the form of 

       - ascii data files
       - png images or
       - gif animations. 

  - the far field power flux of the total field, the initial field or the scattered field. 
    For plane wave excitation, it can be processed to the form of 
    :doc:`differential scattering and extinction cross sections<cross_sections>`.

  - For dipole sources, the dissipated power can be computed (Purcell effect).


Current limitations
---------------------

The following issues need to be considered when applying Smuthi:

  - Particles must not intersect with each other or with layer interfaces.
  - Magnetic or anisotropic materials are currently not supported.
  - The method is in principle valid for a wide range of particle sizes -  
    however, the numerical validity has only been tested for particle diameters up to around one wavelength.
    For larger particles, note that the number of multipole terms in the spherical wave expansion 
    grows with the particle size. For further details, see the 
    :doc:`hints for the selection of the multipole truncation order <simulation_guidelines>`.
  - Smuthi does not provide error checking of user input, nor does it check if 
    numerical parameters specified by the user are sufficient for accurate 
    simulation results. It is thus required that the user develops some 
    understanding of the influence of various numerical parameters on the 
    validity of the results. 
    See the :doc:`simulation guidelines <simulation_guidelines>`.
  - A consequence of using the T-matrix method is that the electric field inside the circumscribing
    sphere of a particle cannot be correctly computed, see for example `Auguié et al. (2016) <https://doi.org/10.1088/2040-8978/18/7/075007>`_. 
    In the electric field plots, the circumscribing sphere is displayed as a dashed circle around the particle
    as a reminder that there, the computed near fields cannot be trusted.
  - Particles with initersecting circumscribing spheres can lead to incorrect results. 
    The use of Smuthi is therefore limited to geometries with particles that have disjoint circumscribing spheres.
  - If particles are located near interfaces, such that the circumscribing shere of the particle intersects the 
    interface, a correct simulation result can in principle be achieved. However, special care has to be taken
    regarding the selection of the truncation of the spherical and plane wave expansion, see
    the :doc:`hints for the selection of the wavenumber truncation<simulation_guidelines>`. 


License
-------

The software is licensed under the `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_.


Contact
---------

Smuthi was written and is maintained by Amos Egel. Email to |emailpic| for questions, feature requests or if you would like to contribute.

.. |emailpic| image:: images/email.png


Acknowledgments
---------------

Smuthi includes contributions from the following persons:

   - Adrian Doicu, Thomas Wriedt and Yuri Eremin through the
     `NFM-DS <https://scattport.org/index.php/programs-menu/t-matrix-codes-menu/239-nfm-ds>`_ package, a copy of which
     is distributed with Smuthi.
   - Dominik Theobald implemented functions for the simulation of particles with arbitrary orientation. 
     He currently works on the implementation of a plane wave based particle coupling for non-spherical particles 
     with overlapping circumscribing spheres.
   - Konstantin Ladutenko with many useful additions, including example simulations, smoother input/output, 
     support of MPI parallel computing (currently in construction) and the option of permanent NFM-DS installation folder.

Big thanks go to Lorenzo Pattelli for designing the Smuthi logo.

Ilia Rasskazov has helped with useful comments and bug reports.

The creation of Smuthi was supervised by Uli Lemmer and Guillaume Gomard during the research project
`LAMBDA <http://gepris.dfg.de/gepris/projekt/278746617>`_, funded by the `DFG <http://www.dfg.de/>`_ 
in the priority programme `tailored disorder <http://gepris.dfg.de/gepris/projekt/255652081>`_.



