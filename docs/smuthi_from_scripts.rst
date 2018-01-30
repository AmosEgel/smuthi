=========================
Run smuthi from scripts
=========================

TBD

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
 

Back to :doc:`main page <index>`
