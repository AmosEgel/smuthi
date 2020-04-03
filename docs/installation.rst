Installation
=============

We **recommend to use Linux operating systems** to run Smuthi. Otherwise, Smuthi can run on Windows, too, but issues regarding dependencies or performance are more likely.

Installing Smuthi under Ubuntu (recommended)
--------------------------------------------
`python3` with `pip3`, `gfortran` and `gcc` usually are shipped with the operating system.

Make sure that the Foreign Function Interface library is available::

  sudo apt-get install libffi6 libffi-dev

Then install Smuthi from PyPi::

  sudo pip3 install smuthi

or locally (see below section :ref:`local_install`).



Installing Smuthi under Windows
-------------------------------

First make sure that Python 3 is installed on your computer. 
You can install for example 
`Anaconda <https://www.continuum.io/downloads>`_ 
or `WinPython <https://winpython.github.io/>`_ 
to get a full Python environment.

Then, open a command window and type::

    pip install smuthi

Depending on where pip will install the package, you might need administrator rights for that.

Alternatively, install locally (see below section :ref:`local_install`).


gfortran under Windows (for NFM-DS, optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note:: 
	Smuthi comes with a precompiled NFM-DS executalbe for Windows. So, if you skip the installation of `gfortran`, you might still be able to use it. However, in some Windows versions, certain DLLs that are required to run the executable might be missing. 
	In that case, Smuthi will throw an error indicating that `gfortran` is missing.

Visit the `MinGW getting started page <http://mingw.org/wiki/Getting_Started>`_ and follow the instructions to install `gfortran`. 
Also make sure to add the bin folder of your MinGW installation to the Windows PATH variable. See `Environment Settings` section of the `MinGW getting started page <http://mingw.org/wiki/Getting_Started>`_ for instructions.

C compiler under Windows (for pywigxjpf, optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note:: 
	If you skip the installation of a C compiler, Smuthi will still work, but you might not benefit from fast evaluation of Wigner3j symbols through the wigxjpf libary. This can be an issue when your simulation involves large multipole degrees.

To benefit from faster evaluation of Wigner3j symbols through the pywigxjpf package, you need a C compiler.
If you have Microsoft Visual Studio installed, `MS VC` is probably already there. Otherwise, open the Visual Studio setup and install the Visual C compiler. If you don't have Microsoft Visual Studio, see 
`the Python Wiki <https://wiki.python.org/moin/WindowsCompilers>`_ 
for further instructions.


.. _GPUAnchor:

GPU-acceleration (optional)
---------------------------
.. note:: 
	PyCuda support is recommended if you run heavy simulations with many particles. In addition, it can speed up certain post processing steps like the evaluation of the electric field on a grid of points, e.g. when you create images of the field distribution. 
	For simple simiulations involving one particle on a substrate, you might well go without.

If you want to benefit from fast simulations on the GPU, you need:

* A CUDA-capable NVIDIA GPU
* The `NVIDIA CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_ installed
* PyCuda installed

Under Ubuntu, install PyCuda simply by::

  sudo pip3 install pycuda

Under Windows, installing PyCuda this is not as straightforward as under Linux.
There exist prebuilt binaries on `Christoph Gohlke's homepage <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda>`_. 
See for example `these instructions <https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_PyCUDA_On_Anaconda_For_Windows?lang=en>`_ 
for the necessary steps to get it running. 


.. _local_install:

Installing Smuthi locally
-------------------------
As an alternative to :code:`pip3 install smuthi` or :code:`pip install smuthi` (which download the latest release from the Python package index, PyPi), you can download the Smuthi project folder manually from `here <https://gitlab.com/AmosEgel/smuthi/tags>`_
or git fork the `gitlab repository <https://gitlab.com/AmosEgel/smuthi.git>`_. Open a command prompt and change directory to the Smuthi
project folder. Then, enter (Windows)::

   pip install .

or (Ubuntu)::

   sudo pip3 install .

If you plan to edit the Smuthi code, install in develop mode by (Windows)::

   pip install -e .

or (Ubuntu)::

   sudo pip3 install -e .

This option allows to install a non-release version of Smuthi or to modify the source code and then run your custom version of Smuthi.


NFM-DS
------
The NFM-DS Fortran package by Doicu, Wriedt and Eremin is shipped together with Smuthi.
Whenever you run a Smuthi simulation containing non-spherical particles,
it will create a copy of the NFM-DS package in a temporary folder and compile it.

To avoid these redundant NFM-DS copies
(and to save the time needed to compile NFM-DS during each Smuthi simulation), 
you can create an empty folder named :code:`smuthi_nfmds_bin`
at your working path to keep the NFM-DS binary between simulations. This folder
will be used by Smuthi for a persistent installation of the NFM-DS package
that is used for all subsequent Smuthi simulations started from that working path.
