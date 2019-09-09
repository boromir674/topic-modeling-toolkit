Topic Modeling Toolkit - Python Library
=========================================================================

This library aims to automate Topic Modeling research-related activities.

* Data preprocessing and dataset computing
* Model training (with parameter grid-search), evaluating and comparing
* Graph building
* Computing KL-divergence between p(c|t) distributions
* Datasets/models/kl-distances reporting


.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |travis|
      - | |coverage|
      - | |code_intelligence|

.. |travis| image:: https://travis-ci.org/boromir674/topic-modeling-toolkit.svg?branch=packagify
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/boromir674/topic-modeling-toolkit

.. |coverage| image:: https://coveralls.io/repos/github/boromir674/topic-modeling-toolkit/badge.svg?branch=packagify
    :alt: Coverage Status
    :target: https://coveralls.io/github/boromir674/topic-modeling-toolkit?branch=packagify

.. |code_intelligence| image:: https://scrutinizer-ci.com/g/boromir674/topic-modeling-toolkit/badges/code-intelligence.svg?b=packagify
    :alt: Code Intelligence
    :target: https://scrutinizer-ci.com/code-intelligence


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
| Please build BigARTM following the instructions `here <https://bigartm.readthedocs.io/en/stable/installation/index.html>`_.
| Make sure to build it in a way to produce the correct wheel to install as bigartm dependency in your enviroment.
| After installing Bigartm, the 'bigartm' executable should be in PATH.
|
| The code shall be hosted on PyPI, hence it should be installed by

::

    $ pip install topic_modeling_toolkit


Usage
-----
A sample example is below.

::

    $ current_dir=$(echo $PWD)
    $ mkdir datasets-dir
    $ export COLLECTIONS_DIR=$current_dir/datasets-dir

    $ transform posts pipeline.cfg my-dataset
    $ train my-dataset train.cfg plsa-model --save
    $ make-graphs --model-labels "plsa-model" --allmetrics --no-legend
    $ xdg-open $COLLECTIONS_DIR/plsa-model/graphs/plsa*prpl*

Citation
--------

1. Vorontsov, K. and Potapenko, A. (2015). `Additive regularization of topic models <http://machinelearning.ru/wiki/images/4/47/Voron14mlj.pdf>`_. Machine Learning, 101(1):303–323.

