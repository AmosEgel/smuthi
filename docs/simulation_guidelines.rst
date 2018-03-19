.. note:: This site is currently under construction.


Some unstructered notes:

To simulate few or the only one particle on the surface you should
switch off the lookup table for particle coupling. The lookup
doesn't work for a single particle (and it is also not recommended for
a small particle numbers in general).  To this end, set
:code:`store_coupling_matrix=True` and
:code:`coupling_matrix_lookup_resolution=None` in
:code:`smuthi.simulation.Simulation()`. Actually, anyone of these settings is
sufficient, as it implies the other.

For small particle numbers it might also make sense to use the :code:`"LU"`
solver instead of :code:`"gmres"`.
 
.. todo:: write hints for lmax

.. todo:: write hints for kmax

`Paper on kmax <https://arxiv.org/abs/1708.05557>`_.

Comment by Kostya:

It is very important to use :code:`l_max` large enough to cover all
significant modes in every particle. However, for large particles with high
refractive index :code:`l_max` should be rather large, this will
need significant compuatational resources. Note that the T-matrix
of a single particle has dimension :math:`D \times D`, where
:math:`D=2(l_{max}+1)2−2`.
