.. _overview:

Overview
========

:Release: |version|
:Date: |today|

Bolt is structured in a number of modules. Each module provides an abstraction of common functionality.



Data Structures
---------------

Bolt itself is constructed around a number of primitive data structures:
  - **Dense vectors**
      implemented via numpy arrays (`numpy.array`).
  - **Sparse vectors**
      implemented via numpy record arrays (`numpy.recarray`). 
 
Each :class:`bolt.model.Model` is parametrized by a weight vector which is represented as a 1D `numpy.array`, allowing efficient random access. 

An instance (aka a feature vector), on the other hand, is represented as a s sparse vector via a numpy record array. The data type of the record array is `bolt.sparsedtype` which is a tuple (uint32,float64). 
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
   :undoc-members:
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
