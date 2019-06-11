Installation
=============
First make sure that Python 3 is installed on your computer. With Linux, this is usually the case. Windows users can
install for example `Anaconda <https://www.continuum.io/downloads>`_ or `WinPython <https://winpython.github.io/>`_ to
get a full Python environment.

Prerequisites
----------
Smuthi uses WIGXJPF to evaluate Wigner symbols, however, it can be a bit tricky to install. Under Linux you need to install CFFI and pycparser::

    sudo apt-get install libffi6 libffi-dev
    sudo pip3 install pycparser
    sudo pip3 install pywigxjpf

Under Windows you will also need some C++ compiler.

.. todo:: check windows integration

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
or git fork the `gitlab repository <https://gitlab.com/AmosEgel/smuthi.git>`_. Open a command prompt and change directory to the Smuthi
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
Whenever you run a Smuthi simulation containing non-spherical particles,
it will create a copy of the NFM-DS Fortran package into a temporary
folder.

Alternatively, you can create an empty folder named :code:`smuthi_nfmds_bin`
at your working path to keep the NFM-DS binary between simulations. This folder
will be used by Smuthi for a persistent installation of the NFM-DS package
that is used for all subsequent Smuthi simulations started from this working path.
