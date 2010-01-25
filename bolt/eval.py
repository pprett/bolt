"""
"""
from __future__ import division

from itertools import izip

import numpy as np
import bolt

def errorrate(model,examples, labels):
    """Compute the misclassification rate of the model.

    zero/one loss: if p*y > 0 then 0 else 1

    Parameters:
    model: An instance of LinearModel

    examples: A sequence of sparse encoded examples.
    lables: A sequence of class labels, either 1 or -1. 
    """
    n = 0
    err = 0
    for p,y in izip(model.predict(examples),labels):
        z = p*y
        if np.isinf(p) or np.isnan(p) or z <= 0:
            err += 1
        n += 1
    errrate = err / n
    return errrate * 100.0

def rmse(model,examples, labels):
    """Compute the root mean squared error of the model.

    Parameters:
    model: An instance of LinearModel

    examples: A sequence of sparse encoded examples.
    lables: A sequence of regression targets.
    """
    n = 0
    err = 0
    for p,y in izip(model.predict(examples),labels):
        err += (p-y)**2.0
        n += 1
    err /= n
    return np.sqrt(err)

def cost(model,examples,labels):
    loss = model.loss
    cost = 0
    for p,y in izip(model.predict(examples),labels):
        cost += loss.loss(p,y)
    print ("cost: %f." % (cost))
        

def error(lm,texamples,tlabels):
    """Report the error of the model on the
    test examples. If the loss function of the model
    is 
    """
    err = 0.0
    if isinstance(lm.loss,bolt.Classification):
        err = errorrate(lm,texamples,tlabels)
        #print("error-rate: %f%%." % (err))
    elif isinstance(lm.loss,bolt.Regression):
        err = rmse(lm,texamples,tlabels)
        #print("rmse: %f." % (err))
    else:
        raise ValueError, "lm.loss: either Regression or Classification loss expected"
    return err
