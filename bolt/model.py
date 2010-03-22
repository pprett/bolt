"""
The :mod:`bolt.model` module contains classes which represent
parametric models supported by Bolt. 

Currently, the following models are supported:

  :class:`bolt.model.LinearModel`: a linear model for binary classification and regression.

"""

__authors__ = [
      '"Peter Prettenhofer" <peter.prettenhofer@gmail.com>'
]

import numpy as np

from io import sparsedtype, dense2sparse

import bolt

class LinearModel(object):
    """A linear model of the form :math:`y = w^T x + b`. 
    """
    
    def __init__(self, m, biasterm = False):
        """Create a linear model with an
        m-dimensional vector `w = [0,..,0]` and `b = 0`.

        :arg m: The dimensionality of the classification problem (i.e. the number of features).
	:type m: positive integer
        :arg biasterm: Whether or not a bias term (aka offset or intercept) is incorporated.
	:type biasterm: True or False
         
        """
        if m <= 0:
            raise ValueError, "Number of dimensions must be larger than 0."
        self.m = m
        self.w = np.zeros((m), dtype=np.float64, order = "c")
        self.bias = 0.0
        self.biasterm = biasterm


    def __call__(self,x):
        """Predicts the target value for the given example. 

	:arg x: An instance in dense or sparse representation. 
        :returns: A float :math:`y = w^T x + b`.
	
        """
        if x.dtype == sparsedtype:
            return bolt.predict(x, self.w, self.bias)
        else:
            sparsex = dense2sparse(x)
            return bolt.predict(sparsex, self.w, self.bias)

    def predict(self,examples):
        """Evaluates :math:`y = w^T x + b` for each
        example x in examples. 

        :arg examples: a sequence of examples.
        :returns: a generator over the predictions.
        """
        for x in examples:
            yield self.__call__(x)
