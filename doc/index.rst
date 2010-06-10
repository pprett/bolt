..  Bolt documentation master file, created by
   sphinx-quickstart on Sat Mar  6 20:41:04 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bolt Online Learning Toolbox
============================

:Release: |version|
:Date: |today|
:Author: peter.prettenhofer@gmail.com

Bolt features discriminative learning of linear predictors (e.g. SVM or
logistic regression) using stochastic gradient descent. Bolt is
aimed at large-scale, high-dimensional and sparse machine-learning problems.
In particular, problems encountered in information retrieval and
natural language processing.

**Contents**:

.. toctree::
   :maxdepth: 1

   install.rst              
   overview.rst                
   using-cli.rst
   using-api.rst
   whatsnew.rst 

Features
========

Currently, Bolt provides the following models for binary and multi-class classification. 

+---------------+----------------+--------------+-----------+
|Model          |Loss            |Penalty       |Multi-class|
+===============+================+==============+===========+
|*SGD*          | - Hinge        | - L2         |OVA        |
|               | - Mod. Huber   | - L1         |           |
|               | - Log          | - ElasticNet |           |
|               |                |              |           |
+---------------+----------------+--------------+-----------+
|*PEGASOS*      | - Hinge        | - L2         |OVA        |
+---------------+----------------+--------------+-----------+
|*AvgPerceptron*| - Zero-One     | - (Averaging)|True       |
+---------------+----------------+--------------+-----------+
|*MaxentSGD*    | - Cross-Entropy| - L2         |True       |
|               |                |              |           |
+---------------+----------------+--------------+-----------+


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

