.. _overview:

Overview
========

:Release: |version|
:Date: |today|

Bolt is structured in a number of modules. Each module provides an abstraction of common functionality. 

  * The :mod:`model` Module
      Parametric models for (multi-class) classification and regression.
  * The :mod:`bolt` Extension Module
      Provides efficient trainers for models. 
  * The :mod:`io` Module
      Input/Output routines and dataset handeling. 
  * The :mod:`eval` Module 
      Function related to evaluation purpose.

Data Structures
---------------

Bolt itself is constructed around a number of primitive data structures:
  - **Dense vectors**
      implemented via numpy arrays (`numpy.array`).
  - **Sparse vectors**
      implemented via numpy record arrays (`numpy.recarray`). 
 
Each `model.Model` is parametrized by a weight vector which is represented as a 1D `numpy.array`, allowing efficient random access. 

An instance (aka a feature vector), on the other hand, is represented as a s sparse vector via a numpy record array. The data type of the record array is `bolt.sparsedtype` which is a tuple (uint32,float64). 
The advantage of record arrays is that they are mapped directly to C struct arrays. 


The :mod:`model` Module
-----------------------

.. automodule:: bolt.model
   :members:
   :show-inheritance:
   :undoc-members:
   :inherited-members:

The :mod:`io` Module
-----------------------

.. automodule:: bolt.io
   :members:
   :show-inheritance:

The :mod:`eval` Module
-----------------------

.. automodule:: bolt.eval
   :members:
   :show-inheritance:
