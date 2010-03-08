.. _using:

Using Bolt
==========

:Release: |version|
:Date: |today|


Binary classification
---------------------

To use Bolt for binary classification we first need some data. Bolt is aimed primarily at high-dimensional and sparse machine learning problems. Thus, instances are represented as sparse vectors. Bolt uses numpy record arrays to represent sparse vectors. The following example shows an instance `x` with features `0` and `4` set to `1` and `0.2`, respectively. All other features are assumed to be zero. ::

  x = np.array([(0,1),(4,0.2)], dtype = np.dtype("u4,f4")

For binary classification, Bolt assumes that the class labels, either positive or negative, are coded as `y = 1` or `y = -1`. 

Lets create some synthetic dataset to show how to use Bolt for binary classification...

For convenience, Bolt offers routines to load datasets in binary or svm^light format via the :mod:io module. ::

  dataset = bolt.io.MemoryDataset.load("train.dat.gz")

The class `MemoryDataset` represents a in-memory dataset. It offers a number of methods to access the dataset via iteration. 

.. class:: MemoryDataset

   The :class:`MemoryDataset` wraps two parallel arrays `instances` and `labels` and controls access to those arrays. In particular, it controls sequential access via the indexing array `self._shuffle`. This allows efficient shuffling of the dataset via the :method:`shuffle` method. 

   The :class:`MemoryDataset` class has the following public methods:

   .. method:: __iter__(self)

      Iterate through the parallel arrays .

   .. method:: shuffle(self, seed = None)

      Shuffles the indexing array. A `seed` for the random number generator can be supplied. 

   