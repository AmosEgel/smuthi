This folder contains source files from the NFM-DS program written by the authors of:

[1] A. Doicu, T. Wriedt, and Y. Eremin: "Light Scattering by Systems of Particles", Berlin, Heidelberg: Springer-Verlag, 2006.

The latest release of NFM-DS can be downloaded from: https://scattport.org

We thank the authors for allowing us to use their routines.

For the purpose of integrating the code for the generation of the T-matrix of axisymmetric particles into the SMUTHI project,
some files were adapted. 

The changes affect:
- explicit include of other modules in file TAXSYM.f90
- replacements of relative paths like "../INPUTFILES/" by path "smuthi/nfmds/data/" and similar
- suppressing the console output of taxsym by commenting out the print commands.

Changes to the original code were highlighted through comments, e.g. like this:

! The following lines were removed from the original code: >>>>>>>>>>>>>>>>>>>>>>>>>>
!lines of original file that were removed (commented out)
!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

! The following lines were added to the original code: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
lines that were added 
!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<