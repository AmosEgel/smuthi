Contributing to Smuthi
=======================
Thanks a lot for your interest to take part in the Smuthi project. Contributions are highly welcome. In order to get in contact with Smuthi's developer community, please consider to `register to the Smuthi mailing list <https://groups.google.com/forum/#!forum/smuthi>`_.

*************************
1 Giving general feedback
*************************
We are grateful to learn about what works well and what doesn't. If you have some user experience to share, please `write to the Smuthi mailing list <mailto:smuthi@googlegroups.com>`_ (preferred) or to the `project owner <mailto:amos.egel@gmail.com>`_.

********************************************
2 Reporting a bug or proposing a new feature
********************************************
Please open a new issue in the Smuthi repository's `issue tracker <https://gitlab.com/AmosEgel/smuthi/issues>`_ to report a bug or to request a new feature.

**Good bug reports** tend to have:

- A quick summary and/or background
- Steps to reproduce (give sample code if you can).
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

*************************************
3 Submitting to the Smuthi repository
*************************************
In general, the workflow for contributing source code, example scripts or documentation to Smuthi is like this:

1. Start from an issue. If the addition or modification that you have in mind is not yet represented by an issue in the `issue tracker`_, create one. This gives the other developers the chance to comment on your idea beforehand.
2. When you commit yourself to address some issue, you can assign it to you.
3. Fork the repository and create a branch from master (please use a meaningful name for the branch that refers to the addition that you plan, like for example `application-example-metasurface` or `cubic-particles` or `bugfix-t-matrix-readout`).
4. Add your code and commit.
5. If you have added code that should be tested, add tests.
6. If you have modified some source code, ensure the test suite passes.
7. If you have modified the API, update the documentation.
8. Create a `merge request <https://gitlab.com/AmosEgel/smuthi/merge_requests>`_.

3.1 Submitting an example script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Smuthi is a complex scientific simulation software, and it is not straightforward to generate meaningful results with it. Examples are a great way to teach using this software properly, as well as to demonstrate the range of possible applications. We also seek to validate our results by comparison to other established software, and to measure the runtime in comparison to that of other software.

In either case, a tutorial, application example or accuracy/performance benchmark consists of

- A Python script that runs some kind of simulation. The script should be as short as possible. Use comments to explain every bit of it to unexperienced users. Your can use the `script of the "sphere on a substrate" tutorial <https://smuthi.readthedocs.io/en/latest/_downloads/dielectric_sphere_on_substrate.py>`_ as a template. Please regard the section about code style (see below).

- If applicable, other data needed to reproduce the results, like

  - results from other software for comparison
  - refractive index data

- A short paragraph discussing the simulation results, see for example the `discussion of the "sphere on a substrate" tutorial <https://smuthi.readthedocs.io/en/latest/examples/sphere_on_substrate/discussion.html>`_.

3.1.1 Tutorials
"""""""""""""""
A tutorial is a short script that explains some aspect of Smuthi to new users. See the `current list of tutorials <https://smuthi.readthedocs.io/en/latest/examples.html>`_. If you consider to contribute a tutorial, first reflect about **what is the aspect that you want to teach the user?** Then build your tutorial around that aspect. If not existing yet, open an issue describing what you plan to do. Assign the issue to you.

Store your tutorial script under the path `examples/tutorials/your_tutorial_name`. If other data is needed to run the script (for example, if the script calls some refractive index data file), please also provide that data and additionally put a zip folder containing both the script and the data files into the same directory.

Put a short article discussing the output of the tutorial script into the documentation folder, `docs/examples/your_tutorial_name`. The article should be written in `RST format <https://de.wikipedia.org/wiki/ReStructuredText>`_.

Update the list of tutorials in the file `docs/examples.rst` and create a link to your article.

3.1.2 Application examples
""""""""""""""""""""""""""
An application example is a demo how Smuthi is used in a real or realistic application scenario. Unlike in a tutorial, the focus is not on teaching how Smuthi works, but rather on showing the range of possible applications. If you have used Smuthi to generate interesting results, you can submit them as an application example. We also encourage you to add a link to any paper using Smuthi. Currently, there are not yet any application examples with Smuthi. If you consider to create one, please contact the `project owner`_ to synchronize.

3.1.3 Accuracy/performance benchmarks
"""""""""""""""""""""""""""""""""""""
An accuracy benchmark is a quantitative comparison between Smuthi results and results from  other software. These examples are important to demonstrate the validity of Smuthi and to build up trust.
If you consider to contribute an accuracy benchmark and if not existing yet, open an issue describing what you plan to do. Assign the issue to you.

Store the Smuthi script generating the results to compare under the path `examples/benchmarks/your_benchmark_name`. Also add the results from the other software in raw data format and if possible, also add some input file or model file to reproduce that data with the other software. In addition, ad a zip archive including everything at the same location.

Put a short article discussing the accuracy of the agreement into the documentation folder, `docs/examples/your_benchmark_name`. The article should be written in `RST format <https://de.wikipedia.org/wiki/ReStructuredText>`_.

If you are able to measure the runtime of both Smuthi and the other software under fair conditions, you can also add these data to your discussion.

Update the list of benchmarks in the file `docs/examples.rst` and create a link to your article.

3.2 Submitting source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
You are invited to modify or add to the source code of Smuthi. Although it is a rather complex scientific software, in many cases you can contribute by improving some unit without the need to fully understand what the whole software does in every detail. So, if you feel motivated to address some issue in the `issue tracker`_, go for it! The more experienced members of the Smuthi develpers community will be glad to assist you when you encounter some difficulty. Just `write to the Smuthi mailing list`_.

When you add or modify source code to Smuthi, make sure to

- adhere to our code style (see below)
- check if any functionality was broken by running the test suite (see below)
- if you add functionality, write a test for it, too

3.2.1 Submitting a bug fix or other code improvements
"""""""""""""""""""""""""""""""""""""""""""""""""""""
If you found a bug in the source code, we would be glad if you submit a pull request fixing it. 

Other code improvements could for example lead to a speedup of some unit or make the code more readable. In that case, please add a small test demonstratig that the improved code section behaves the same as before your modification (if not obvious). 

3.2.2 Submitting a new feature
""""""""""""""""""""""""""""""
The Smuthi project is work in progress, and the range of functionality has not even come close to its potential amplitude. Pull requests offering a new feature are always highly welcome.
Please always start from an issue in the `issue tracker`_ in order to give other developers the chance to comment on your addition beforehand. If your addition is a major and entirely new feature, consider to implement it as a subpackage (i.e., store your code into a subdirectory of the smuthi directory).

Typical new features could be ...

- new post processing routines (to process simulation results into some quantity or image)
- new initial fields (e.g. new beam shapes)
- new functions to compute the single particle T-matrices
- (connected with the last:) new particle classes
- any other new feature

To do: Explain how to implement a new T-matrix interface

************
4 Code style
************
Please refer to the `PEP 8 Python code style guide <https://www.python.org/dev/peps/pep-0008/>`_. In Smuthi, we use `snake_case` for variable names, function names and module names, and `CamelCase` for class names. We accept long lines usage (up to 120 symbols).

***************
5 Running tests
***************
You can use `nosetests <https://nose.readthedocs.io>`_ to run our test suite.
Under Ubuntu, install it via :code:`pip3 install nose` and run all tests by :code:`nosetests3`.

*********
6 License
*********
By contributing, you agree that your contributions will be licensed under the MIT License.

************
6 References
************
This document was created by adapting `this template <https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62>`_.
