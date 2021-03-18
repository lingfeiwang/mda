=========
MDA
=========
.. image:: https://zenodo.org/badge/301879307.svg
   :target: https://zenodo.org/badge/latestdoi/301879307


MDA (Maximal Discriminating Axes) is a dimensional reduction method to contrast different groups of samples. MDA assumes a normally distributed noise and uses classification training error in Linear Discriminant Analysis (LDA) to estimate the similarity between groups of samples. MDA has been used to compare cell groups in single-cell RNA sequencing.

MDA is a Python3 library and provides examples in Jupyter notebooks. You can read more about the MDA method from manuscript (See References_).

Installation
=============
MDA can be installed with pip: ``python -m pip install git+https://github.com/lingfeiwang/mda.git``. 

Documentation
=============
Documentations are available as `html <https://lingfeiwang.github.io/mda/index.html>`_ and `pdf <https://github.com/lingfeiwang/mda/raw/master/docs/build/latex/mda.pdf>`_.

Examples
==========================
You can find an examples with simulated data in the 'examples' folder.

Contact
==========================
Please raise an issue on `github <https://github.com/lingfeiwang/mda/issues/new>`_ .

References
==========================
* Kwontae You, Lingfei Wang, Chih-Hung Chou, et al. `QRICH1 dictates the outcome of ER stress through transcriptional control of proteostasis <https://science.sciencemag.org/content/371/6524/eabb6896.abstract>`_, Science 2021.
