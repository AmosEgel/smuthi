#!/usr/bin/env bash
# rm -rf env
python3 -m venv env
source env/bin/activate
pip install numpy scipy sympy numba tqdm matplotlib imageio
# pip install pywigxjpf
python setup.py develop