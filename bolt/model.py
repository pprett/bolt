import numpy as np

from io import sparsedtype, dense2sparse

import bolt

class LinearModel(object):
    """A linear model: y = x*w + b. 
    """

    def __init__(self, m, loss = bolt.ModifiedHuber(),
                 reg = 0.001, alpha = 1.0,
                 norm = 2, biasterm = True):
        """Create a linear model with an
        m-dimensional vector w = [0,..,0] and b = 0.

        Parameters:
        m: The dimensionality of the classification problem (i.e. the number of features).
        loss: The loss function (default ModifiedHuber)
        reg: The regularization parameter lambda.
        alpha: The elastic net hyper-paramter alpha. Blends L2 and L1 norm regularization (default 1.0). 
        """
        if m <= 0:
            raise ValueError, "Number of dimensions must be larger than 0."
        if loss == None:
            raise ValueError, "Loss function must not be None."
        if reg < 0.0:
            raise ValueError, "reg must be larger than 0. "
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError, "alpha must be in [0,1]."
        self.m = m
        self.loss = loss
        self.reg = reg
        self.alpha = alpha
        self.w = np.zeros((m),dtype=np.float64)
        self.bias = 0.0
        self.norm = norm
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
