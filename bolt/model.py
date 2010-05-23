"""
The :mod:`bolt.model` module contains classes which represent
parametric models supported by Bolt. 

Currently, the following models are supported:

  :class:`bolt.model.LinearModel`: a linear model for binary classification and regression.
  :class:`bolt.model.GeneralizedLinearModel`: a linear model for multi-class classification.

"""

__authors__ = [
      '"Peter Prettenhofer" <peter.prettenhofer@gmail.com>'
]

import numpy as np

from io import sparsedtype, densedtype, dense2sparse

try:
    from trainer.sgd import predict
except ImportError:
    def predict(x, w, b):
        return np.dot(x, w) + b

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
        """The number of features. 
        """
        self.w = np.zeros((m), dtype=np.float64, order = "c")
        """A vector of size `m` which parameterizes the model. """
        self.bias = 0.0
        """The value of the bias term."""
        self.biasterm = biasterm
        """Whether or not the biasterm is used."""


    def __call__(self, x, confidence = False):
        """Predicts the target value for the given example. 

	:arg x: An instance in dense or sparse representation.
        :arg confidence: whether to output confidence scores.
        :returns: The class assignment :math:`sign(w^T x + b)`
        and optionally a confidence score.
	
        """
        if x.dtype == densedtype:
            x = dense2sparse(x)
        p = predict(x, self.w, self.bias)
        if confidence:
            return np.sign(p), 1.0/(1.0+np.exp(-p))
        else:
            return np.sign(p)

    def predict(self,examples, confidence = False):
        """Evaluates :math:`y = sign(w^T x + b)` for each
        example x in examples.
        See :meth:LinearModel.__call__ .

        :arg examples: a sequence of examples.
        :arg confidence: whether to output confidence scores.
        :returns: a generator over the predictions.
        """
        for x in examples:
            yield self.__call__(x, confidence)

class GeneralizedLinearModel(object):
    """A generalized linear model of the form :math:`z = argmax_y w^T f(x,y) + b_y`.
    """

    def __init__(self, m, k, biasterm = False):
        """Create a generalized linear model for
	classification problems with `k` classes. 

        Parameters:
        m: The dimensionality of the input data (i.e., the number of features).
        k: The number of classes.
        """
        if m <= 0:
            raise ValueError, "Number of dimensions must be larger than 0."
        if k <= 1:
            raise ValueError, "Number of classes must be larger than 2 (if 2 use `LinearModel`.)"
        self.m = m
        """The number of features."""
        self.k = k
        """The number of classes."""
        self.W = np.zeros((k,m), dtype=np.float64, order = "c")
        """A matrix which contains a `m`-dimensional weight vector for each
        class. """
        self.biasterm = biasterm
        self.b = np.zeros((k,), dtype=np.float64, order = "c")
        """A vector of bias terms. """


    def __call__(self,x, confidence = False):
        """Predicts the class for the instance `x`.
        Evaluates :math:`z = argmax_y w^T f(x,y) + b_y`.

        :arg confidence: whether to output confidence scores.
        
        """
        return self._predict(x, confidence)
            

    def predict(self, instances, confidence = False):
        """Predicts class of each instances in
        `instances`.
        See :meth:`GeneralizedLinearModel.__call__`.

        Parameters:
        :arg confidence: whether to output confidence scores.
        :arg examples: a sequence of instances

        :return: a generator over the predictions.
        """
        for x in instances:
            yield self.__call__(x, confidence)

    def _predict(self, x, confidence = False):
        ps = np.array([predict(x, self.W[i], self.b[i]) for i in range(self.k)])
	c = np.argmax(ps)
        if confidence:
            return c,ps[c]
        else:
            return c

    def probdist(self,x):
        """The probability distribution of class assignment.
        """
        ps = np.array([np.exp(predict(x, self.W[i], self.b[i])) for i in range(self.k)])
        Z = np.sum(ps)
        return ps / Z

