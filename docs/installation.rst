Installation
=============
First make sure that Python 3 is installed on your computer. On Linux, this is usually the case. Windows users can use
for example `Anaconda <https://www.continuum.io/downloads>`_ or `WinPython <https://winpython.github.io/>`_ to get a
full python environment. Using pip, you can then install SMUTHI simply by (Windows)::

    pip install smuthi

Depending on where pip will install the package, you might need administrator rights for that. Under (Ubuntu), type::

   sudo pip3 install smuthi

Alternatively, you can download the SMUTHI project folder manually from `here <https://gitlab.com/AmosEgel/smuthi/tags>`_.
Open a command prompt and change directory to the SMUTHI project folder. Then, enter (Windows)::

   python setup.py install

or (Ubuntu)::

   sudo python3 setup.py install

When you run a simulation (containing non-spherical particles) for the first time after installation, you will be asked
to enter a path where SMUTHI will install the NFM-DS Fortran package. This automatically created folder should not be
removed or modified afterwards. Otherwise, the simulation of non-spherical particles becomes impossible and you might
need to re-install SMUTHI.
