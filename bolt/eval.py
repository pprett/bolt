"""
Evaluation
==========

This module contains various routines for model evaluation.

Metrics
-------

- `errorrate`, the error rate of the binary classifier.
- `rmse`, the root mean squared error of a regressor.
- `cost`, the cost of a model w.r.t. a given loss function.
- `error`, the error (either errorrate or rmse) of the model.

"""
from __future__ import division

from itertools import izip

import numpy as np
import bolt

def errorrate(model,ds):
    """Compute the misclassification rate of the model.
    Assumes that labels are coded as 1 or -1. 

    zero/one loss: if p*y > 0 then 0 else 1

    Parameters:
    model: A `LinearModel`
    ds: A `Dataset`
    """
    n = 0
    err = 0
    for p,y in izip(model.predict(ds.iterinstances()),ds.iterlabels()):
        z = p*y
        if np.isinf(p) or np.isnan(p) or z <= 0:
            err += 1
        n += 1
    errrate = err / n
    return errrate * 100.0

def rmse(model,ds):
    """Compute the root mean squared error of the model.

    Parameters:
    model: A `LinearModel`
    ds: A `Dataset`

    Return:
    """
    n = 0
    err = 0
    for p,y in izip(model.predict(ds.iterinstances()),ds.iterlabels()):
        err += (p-y)**2.0
        n += 1
    err /= n
    return np.sqrt(err)

def cost(model,ds, loss):
    """The cost of the loss function.

    Parameters:
    model: A `LinearModel`
    ds: A `Dataset`
    """
    cost = 0
    for p,y in izip(model.predict(ds.iterinstances()),ds.iterlabels()):
        cost += loss.loss(p,y)
    print ("cost: %f." % (cost))

def error(model,ds):
    """Report the error of the model on the
    test examples. If the loss function of the model
    is

    Parameters:
    model: A `LinearModel`
    ds: A `Dataset`
    """
    err = 0.0
    if isinstance(model.loss,bolt.Classification):
        err = errorrate(model,ds)
    elif isinstance(model.loss,bolt.Regression):
        err = rmse(model,ds)
    else:
        raise ValueError, "lm.loss: either Regression or Classification loss expected"
    return err
