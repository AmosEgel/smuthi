Running a simulation
======================
There are two ways to call smuthi:

   * From a Python script. This option is more flexible regarding how to run and evaluate the simulations, and is thus the recommended way to run a simulation.

   * From the command line with an input file. No programming erperience required.

In order to avoid incorrect simulation results or unnecessarily high 
computational effort, please consider the 
:doc:`simulation guidelines <simulation_guidelines>`.


Python scripting
-----------------
In the :doc:`examples <examples>` section you can find a number of example 
scripts that illustrate the use of Smuthi. Edit and run these scripts to get a 
quick start.

For furhter details, the :doc:`API section <smuthi_api>` contains a description
of all of Smuthi's modules, classes and functions.


Call from command line
-----------------------
Alternatively, Smuthi can be executed from the command line together with one 
argument, specifying the input file that contains the parameters of the 
configuration to be simulated.

Open a command window (shell or Win Python Command Prompt) and type::

   smuthi path/to/input.dat

If :code:`smuthi` is called without an argument, it uses an
:download:`example_input.dat <../smuthi/data/example_input.dat>`. 
The output should look like this:

.. todo:: replace screenshot by current version

.. image:: images/console_screenshot.png


Input files
~~~~~~~~~~~~
When Smuthi is called from the command line, the input file uses the 
`YAML <http://yaml.org/>`_ format.
Download an example file 
:download:`example_input.dat <../smuthi/data/example_input.dat>` 
and play around with its entries to get a quick start.

For a detailed explanation of the specified parameters, see the 
:doc:`section on input files <input_files>`.
