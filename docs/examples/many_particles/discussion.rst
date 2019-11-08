Many spheres on a substrate
===========================

This tutorial demonstrates how to set up a simulation containing many particles.

Configuration
-------------
The configuration under study consists of a number of dielectric spheres that are
arranged in the shape of a spiral on a glass substrate.

.. image:: vogel_spiral_200.png
   :width: 45%

.. image:: drawing.png
   :width: 45%

The spheres are illuminated by a plane wave from top under normal incidence.

How to compute large systems
----------------------------
In order to limit the runtime, Smuthi offers various numerical strategies for
the solution of the scattering problem.

To do: complete the discussion

.. image:: cross_section.png
   :width: 75%
   :align: center

.. image:: runtime.png
   :width: 75%
   :align: center

Exemplary far field results
---------------------------

.. image:: dscs_200spheres_top.png
   :scale: 50%

.. image:: dscs_200spheres_bottom.png
   :scale: 50%

The above images show the resulting differential scattering cross section for a spiral of 200 spheres in the top hemisphere
(reflection) and in the bottom hemisphere (transmission).


