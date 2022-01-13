Topic Modeling Toolkit - Python Library
=========================================================================

This library aims to automate Topic Modeling research-related activities,
by providing an easy-to-use cli.

The cli provides commands for operations related to:

* Data preprocessing and dataset computing
* Model training (with parameter grid-search), evaluating and comparing
* Graph building
* Computing KL-divergence between p(c|t) distributions
* Datasets/models/kl-distances reporting

The source code of this project has been used for developing the novelty model
in the "Political Spectrum Aware Topic Model" Thesis Project by Konstantinos Lampridis.

The code should also be used in case you would want to reproduce the experimental results reported in the thesis.

The thesis was part of the Artificial Intelligence MSc programme of the University of Amsterdam. 

You can download and read the complete thesis `here <https://scripties.uba.uva.nl/document/676033>`_.


.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |travis|
        | |coverage|
        | |scrutinizer_code_quality|
        | |code_intelligence|
    * - package
      - |version| |python_versions|

.. |travis| image:: https://travis-ci.org/boromir674/topic-modeling-toolkit.svg?branch=dev
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/boromir674/topic-modeling-toolkit

.. |coverage| image:: https://img.shields.io/codecov/c/github/boromir674/topic-modeling-toolkit/dev?style=flat-square
    :alt: Coverage Status
    :target: https://codecov.io/gh/boromir674/topic-modeling-toolkit/branch/dev

.. |scrutinizer_code_quality| image:: https://scrutinizer-ci.com/g/boromir674/topic-modeling-toolkit/badges/quality-score.png?b=dev
    :alt: Code Quality
    :target: https://scrutinizer-ci.com/g/boromir674/topic-modeling-toolkit/?branch=dev

.. |code_intelligence| image:: https://scrutinizer-ci.com/g/boromir674/topic-modeling-toolkit/badges/code-intelligence.svg?b=dev
    :alt: Code Intelligence
    :target: https://scrutinizer-ci.com/code-intelligence

.. |version| image:: https://img.shields.io/pypi/v/topic-modeling-toolkit.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/topic-modeling-toolkit

.. |python_versions| image:: https://img.shields.io/pypi/pyversions/topic-modeling-toolkit.svg
    :alt: Supported versions
    :target: https://pypi.org/project/topic-modeling-toolkit


========
Overview
========

This library serves as a higher level API around the BigARTM_ (artm python interface) library and exposes it conviniently through the command line.

Key features of the Library:

* Flexible preprocessing pipelines
* Optimization of classification scheme with an evolutionary algorithm
* Fast model inference with parallel/multicore execution
* Persisting of models and experimental results
* Visualization

.. _BigARTM: https://github.com/bigartm


Installation
------------
| The Topic Modeling Toolkit depends on the BigARTM C++ library. Therefore first you should first build and install it
| either by following the instructions `here <https://bigartm.readthedocs.io/en/stable/installation/index.html>`_ or by using
| the 'build_artm.sh' script provided. For example, for python3 you can use the following

::

    $ git clone https://github.com/boromir674/topic-modeling-toolkit.git
    $ chmod +x topic-modeling-toolkit/build_artm.sh
    $ # build and install BigARTM library in /usr/local and create python3 wheel
    $ topic-modeling-toolkit/build_artm.sh
    $ ls bigartm/build/python/bigartm*.whl

| Now you should have the 'bigartm' executable in PATH and you can find a built python wheel in 'bigartm/build/python/'
| You should install the wheel in your environment, for example with command

::

    python -m pip install bigartm/build/python/path-python-wheel

| You can install the package with the following command
| When the package gets hosted on PyPI, it should be installed

::

    $ cd topic-modeling-toolkit
    $ pip install .

If the above fails try again including manual installation of dependencies

::

    $ cd topic-modeling-toolkit
    $ pip install -r requirements.txt
    $ pip install .


Usage
-----
A sample example is below.

::

    $ current_dir=$(echo $PWD)
    $ export COLLECTIONS_DIR=$current_dir/datasets-dir
    $ mkdir $COLLECTIONS_DIR

    $ transform posts pipeline.cfg my-dataset
    $ train my-dataset train.cfg plsa-model --save
    $ make-graphs --model-labels "plsa-model" --allmetrics --no-legend
    $ xdg-open $COLLECTIONS_DIR/plsa-model/graphs/plsa*prpl*

Citation
--------

1. Vorontsov, K. and Potapenko, A. (2015). `Additive regularization of topic models <http://machinelearning.ru/wiki/images/4/47/Voron14mlj.pdf>`_. Machine Learning, 101(1):303–323.
