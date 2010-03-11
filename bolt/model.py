"""
Model Specifications
===========================

Classes:

- `LinearModel`, a linear model for binary classification.

"""
import numpy as np

from io import sparsedtype, dense2sparse

import bolt

class LinearModel(object):
    """A linear model of the form y = x*w + b. 
    """

    
    def __init__(self, m, biasterm = True):
        """Create a linear model with an
        m-dimensional vector w = [0,..,0] and b = 0.

        Parameters:
        m: The dimensionality of the classification problem (i.e. the number of features).
         
        """
        if m <= 0:
            raise ValueError, "Number of dimensions must be larger than 0."
        self.m = m
        self.w = np.zeros((m), dtype=np.float64, order = "c")
        self.bias = 0.0
        self.biasterm = biasterm


    def __call__(self,x):
        """Predicts the target value for the given example. 
        
        Return:
        -------
        y = x*w + b
        """
        if x.dtype == sparsedtype:
            return bolt.predict(x, self.w, self.bias)
        else:
            sparsex = dense2sparse(x)
            return bolt.predict(sparsex, self.w, self.bias)

    def predict(self,examples):
        """Evaluates y = x*w + b for each
        example x in examples. 

        Parameters:
        examples: a sequence of examples.

        Return:
        A generator over the predictions.
        """
        for x in examples:
            yield self.__call__(x)
