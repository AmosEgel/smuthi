Installation
=============
First make sure that Python 3 is installed on your computer. With Linux, this is usually the case. Windows users can
install for example `Anaconda <https://www.continuum.io/downloads>`_ or `WinPython <https://winpython.github.io/>`_ to
get a full Python environment.

Using pip
----------
Under Windows, open a command window and type::

    pip install smuthi

Depending on where pip will install the package, you might need administrator rights for that.

Under Ubuntu, type::

   sudo pip3 install smuthi

Installing manually
--------------------
Alternatively, you can download the Smuthi project folder manually from `here <https://gitlab.com/AmosEgel/smuthi/tags>`_
or git fork `<https://gitlab.com/AmosEgel/smuthi.git>`_. Open a command prompt and change directory to the Smuthi
project folder. Then, enter (Windows)::

   python setup.py install

or (Ubuntu)::

   sudo python3 setup.py install

If you plan to edit the Smuthi code, install in develop mode by (Windows)::

   python setup.py develop

or (Ubuntu)::

   python3 setup.py develop


NFM-DS
-------
When you run a Smuthi simulation (containing non-spherical particles) for the first time after installation, you will be asked
to enter a path where it will install the NFM-DS Fortran package. This automatically created folder should not be
removed or modified afterwards. Otherwise, the simulation of non-spherical particles becomes impossible and you might
need to re-install Smuthi.
