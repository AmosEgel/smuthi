Call Smuthi from command line (deprecated)
==========================================
Alternatively, Smuthi can be executed from the command line together with one 
argument, specifying the input file that contains the parameters of the 
configuration to be simulated.

Open a command window (shell or Win Python Command Prompt) and type::

   smuthi path/to/input.dat

If :code:`smuthi` is called without an argument, it uses an
:download:`example_input.dat <../smuthi/_data/example_input.dat>`. 
The output should look like this:

Input files
~~~~~~~~~~~~
When Smuthi is called from the command line, the input file uses the 
`YAML <http://yaml.org/>`_ format.
Download an example file 
:download:`example_input.dat <../smuthi/_data/example_input.dat>` 
and play around with its entries to get a quick start.

For a detailed explanation of the specified parameters, see the 
:doc:`section on input files <input_files>`.
