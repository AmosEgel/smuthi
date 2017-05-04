.. SMUTHI documentation master file, created by
   sphinx-quickstart on Wed May  3 19:50:09 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   
Welcome to SMUTHI's documentation!
==================================
SMUTHI means 'Scattering by MUltiple particles in THIn-film systems'. 
The software allows you to simulate light scattering
by multiple particles near or inside a system of planar layer interfaces. 
It is based on the T-matrix method for the single particle scattering, 
and on the scattering-matrix method for the propagation through the layered medium.
  
SMUTHI is implemented in Python 3.5. 
To run the program, you first need to make sure that Python 3 is installed on your computer.
If this is not the case, see :doc:`here<how_install_python>`.

There are two different ways to run simulations:

   * From the command line with an input file. No programming skills are required. 
	 
   * From a Python script. This is more flexible how to run and evaluate the simulations.
     
Run from command line
=================================

Installing SMUTHI
~~~~~~~~~~~~~~~~~
To run from the command line, SMUTHI needs to be installed. Using pip, you can do that simply by::

   pip install smuthi

Alternatively, you can download the SMUTHI project folder manually from `here <https://gitlab.com/AmosEgel/smuthi/tags>`_. 
Open a command prompt and change directory to the SMUTHI project folder. Then, enter (Windows)::

   python setup.py install

or (Ubuntu)::

   python3 setup.py install

SMUTHI is executed from the command line together with one argument, 
specifying the input file that contains all parameters of the configuration to be simulated, 
see `The input file`_.

Executing SMUTHI
~~~~~~~~~~~~~~~~~
Open a command window (shell or Win Python Command Prompt) and type::

   smuthi path/to/input.dat

If :code:`smuthi` is called without an argument, it tries to open :code:`input.dat` as default in the local folder.
   
The input file
~~~~~~~~~~~~~~~
The input file uses the `YAML <http://yaml.org/>`_ format. 
Download an example file :download:`input.dat <../input.dat>` and play around with its entries to get a quick start.

For a detailed explanation of the specified parameters, see the :doc:`section on input files <input_files>`.



Running simulations as Python scripts
=====================================
When running SMUTHI from Python using scripts, it is not necessary to install it. 
Just download the latest version from `here <https://gitlab.com/AmosEgel/smuthi/tags>`_ and extract the archive.

In the SMUTHI project folder, you find a script called :download:`run_smuthi_as_script.py <../run_smuthi_as_script.py>`.
You can also download it from here by clicking on the above filename.

Edit and run that script to get a quick start.
For details, see the :doc:`section on running SMUTHI from scripts <smuthi_from_scripts>`.

