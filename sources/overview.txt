.. _overview:

API Overview
============

:Release: |version|
:Date: |today|

This section gives an overview of Bolt's API. Bolt itself is designed around a number of 
simple data structures for representing models and input examples. 
The functionality of Bolt is structured in a number of core modules. Each module provides an abstraction of common functionality. For example there exist modules to represent models, train models, load data sets, or evaluate models. 

.. contents::



Data Structures
---------------

Bolt itself is constructed around a number of primitive data structures:
  - **Dense vectors**
      implemented via numpy arrays (`numpy.array`).
  - **Sparse vectors**
      implemented via numpy record arrays (`numpy.recarray`). 
 
Each :class:`bolt.model.Model` is parametrized by a weight vector which is represented as a 1D `numpy.array`, allowing efficient random access. 

An instance (aka a feature vector), on the other hand, is represented as a s sparse vector via a numpy record array. The data type of the record array is data type `bolt.sparsedtype` which is a tuple (uint32,float64). 
The advantage of record arrays is that they are mapped directly to C struct arrays. 


The :mod:`bolt.model` Module
----------------------------

.. automodule:: bolt.model
   :members:
   :show-inheritance:
   :undoc-members:
   :inherited-members:

The :mod:`bolt.trainer` Module
------------------------------

.. automodule:: bolt.trainer
   :members:
   :show-inheritance:
   :undoc-members:
   :inherited-members:

The :mod:`bolt.trainer.sgd` Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: bolt.trainer.sgd
   :members:
   :show-inheritance:
   :inherited-members:

The :mod:`bolt.io` Module
-------------------------

.. automodule:: bolt.io
   :members:
   :show-inheritance:

The :mod:`bolt.eval` Module
---------------------------

.. automodule:: bolt.eval
   :members:
   :show-inheritance:

References
----------

.. [Shwartz2007] Shwartz, S. S., Singer, Y., and Srebro, N., *Pegasos: Primal estimated sub-gradient solver for svm*. In Proceedings of ICML '07.

.. [Tsuruoka2009] Tsuruoka, Y., Tsujii, J., and Ananiadou, S., *Stochastic gradient descent training for l1-regularized log-linear models with cumulative penalty*. In Proceedings of the AFNLP/ACL '09.

.. [Zhang2004] Zhang, T., *Solving large scale linear prediction problems using stochastic gradient descent algorithms*. In Proceedings of ICML '04.

.. [Zou2005] Zou, H., and Hastie, T., *Regularization and variable selection via the elastic net*. Journal of the Royal Statistical Society Series B, 67 (2), 301-320.