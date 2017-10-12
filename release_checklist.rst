releasing a SMUTHI version
===========================

- run nosetests
- update version number and dependencies in setup.py
- update README.rst
- update version number in docs/
- update docs/input_files.rst
- update docs/about_smuthi.rst
- build source distribution by running the command 'python setup.py sdist bdist_wheel' (in the command window)
- upload to PyPi by running the command 'twine upload dist/*' (in the command window)
- push everything to the online repository
- update documentation by building from readthedocs.org

