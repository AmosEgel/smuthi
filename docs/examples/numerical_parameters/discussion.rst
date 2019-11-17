Numerical parameters
====================

In this article we want to explain the meaning of the numerical parameters
controlling Smuthi's performance regarding accuracy and runtime:

 - The multipole cut-off parameters for each scattering particle

 - The Sommerfeld integral contour

 - The particle coupling lookup resolution and interpolation order (if applicable)

 - The discretization and cut-off for the plane wave expansion for the near field calculation

 - The discretization of the far field in direction space

Besides bare guessing, there are two strategies for the setting of numerical parameters: empirical tests and rules of thumb.

Empirical tests check the consistency of simulation results for a given numerical setting.
By "consistency" we mean for example the agreement to accurate benchmark results that can be
analytical results, results from other software or Smuthi results for a more accurate setting.

In certain cases, we can also check how accurately energy is conserved as a consistency criterion.
However, this criterion is suited only for certain numerical parameters.
In other cases, it is misleading.

.. note::
  In certain cases, even inaccurate simulations yield an accurately conserved energy. This will happen for
  example in case of a too small multipole truncation.

.. note::
   Smuthi currently supports only the evaluation of optical power in the far field and for dipole sources,
   which allows to check the conservation of energy only in systems with no absorbing materials and no waveguiding.

Rules of thumb on the other hand can stem from heuristical reasoning or represent former experience.
They can be fit formulae to earlier results from empirical tests,
see for example [Wiscombe] or [Neves] for the selection of multipole truncation
or [Egel2017] for the truncation of Sommerfeld integrals.

In the following, we will discuss the meaning of each numerical parameter, give recommendations
on how to select a reasonable value and demonstrate their influece with a numerical example.

Multipole cut-off
-----------------
The scattering properties of each particle are represented by its T-matrix :math:`T_{plm,p'l'm'}`
where :math:`plm` and :math:`p'l'm'` are the multipole polarization, degree and order of the scattered
and incoming field, respectively, see sections 3.3 and 2.3.2 of |diss|.
In practice, the T-matrix is truncated at some multipole degree :math:`l_{max} \ge 1` and order
:math:`0 \le m_{max} \le l_{max}` to obtain a finite system of linear equations.
In general we can say:

 - Large particles require higher multipole orders than small particles.

 - Particles very close to each other, very close to an interface or very close to a point dipole
   source require higher multipole orders than those that stand freely.

Larger multipole cutoff parameters imply better accuracy, but also a quickly growing numerical effort.

Literature offers various rules of thumb for the selection of the multipole truncation in the
case of spherical particles. The best known is probably the [Wiscombe criterion]



.. |diss| replace:: `[Egel 2018] <https://publikationen.bibliothek.kit.edu/1000093961/26467128>`_