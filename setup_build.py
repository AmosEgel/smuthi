# -*- coding: utf-8 -*-
import sys
from cx_Freeze import setup, Executable

import os

os.environ['TCL_LIBRARY'] = "C:\\Program Files\\Anaconda3\\tcl\\tcl8.6"
os.environ['TK_LIBRARY'] = "C:\\Program Files\\Anaconda3\\tcl\\tk8.6"

#addtional_mods = ['numpy.core._methods', 'numpy.lib.format']
build_exe_options = {'includes': ['numpy.core._methods', 'numpy.lib.format', 'matplotlib']}

base = "Console"
#if sys.platform == "win32":
#    base = "Win32GUI"

setup(
    name="SMUTHI",
    version="0.1",
    author="Amos Egel",
    author_email="amos.egel@kit.edu",
    description="SMUTHI stands for 'Scattering by MUltiple particles in THIn-film systems'.",
    long_description=("SMUTHI is based on the T-matrix method for the single particle scattering, and on the "
                      "scattering-matrix method for the propagation through the layered medium."),
    options={"build_exe": build_exe_options},
    executables=[Executable("smuthi/__main__.py", base=base)]
)
