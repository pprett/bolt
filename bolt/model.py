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

from io import sparsedtype, densedtype, dense2sparse
from trainer import sgd

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
        if x.dtype == densedtype:
            x = dense2sparse(x)
        return sgd.predict(x, self.w, self.bias)

    def predict(self,examples):
        """Evaluates :math:`y = w^T x + b` for each
        example x in examples. 

        :arg examples: a sequence of examples.
        :returns: a generator over the predictions.
        """
        for x in examples:
            yield self.__call__(x)

class GeneralizedLinearModel(object):
    """A generalized linear model of the form z = max_y w * f(x,y) + b_y.
    """

    def __init__(self, m, k):
        """Create a generalized linear model for
	classification problems with `k` classes. 

        Parameters:
        m: The dimensionality of the input data (i.e., the number of features).
        k: The number of classes.
        """
        if m <= 0:
            raise ValueError, "Number of dimensions must be larger than 0."
        if k <= 2:
            raise ValueError, "Number of classes must be larger than 2 (if 2 use `LinearModel`.)"
        self.m = m
        self.k = k
        self.W = np.zeros((k,m), dtype=np.float64, order = "c")
        self.b = np.zeros((k,), dtype=np.float64, order = "c")


    def __call__(self,x):
        """Predicts the class for the instance `x`. 
        
        Return:
        -------
        The class idx. 
        """
        return self._predict(x)
            

    def predict(self,instances):
        """Predicts class of each instances in
        `instances`.

        Parameters:
        examples: a sequence of instances

        Return:
        A generator over the predictions.
        """
        for x in instances:
            yield self.__call__(x)

    def _predict(self,x):
        ps = np.array([sgd.predict(x, self.W[i], self.b[i]) for i in range(self.k)])
	c = np.argmax(ps)
        return c

