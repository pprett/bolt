# encoding: utf-8
# cython: profile=True
# filename: bolt.pyx

from __future__ import division

import numpy as np
import sys

cimport numpy as np
cimport cython

from time import time
from itertools import izip

loss_functions = {0:Hinge, 1:ModifiedHuber, 2:Log, 5:SquaredError, 6:Huber}

cdef extern from "math.h":
    cdef extern double exp(double x)
    cdef extern double log(double x)
    cdef extern double sqrt(double x)

# ----------------------------------------
# Extension Types for Loss Functions
# ----------------------------------------

cdef class LossFunction:
    """Base class for convex loss functions"""
    cpdef double loss(self,p, y):
        raise NotImplementedError()
    cpdef double dloss(self,p, y):
        raise NotImplementedError()

cdef class Regression(LossFunction):
    """Base class for loss functions for regression."""
    cpdef double loss(self,p, y):
        raise NotImplementedError()
    cpdef double dloss(self,p, y):
        raise NotImplementedError()


cdef class Classification(LossFunction):
    """Base class for loss functions for classification."""
    cpdef double loss(self,p, y):
        raise NotImplementedError()
    cpdef double dloss(self,p, y):
        raise NotImplementedError()

cdef class ModifiedHuber(Classification):
    """Modified Huber loss function for binary
    classification tasks with y in {-1,1}.
    Its equivalent to quadratically smoothed SVM
    with gamma = 2. See T. Zhang 'Solving
    Large Scale Linear Prediction Problems Using
    Stochastic Gradient Descent', ICML'04.
    """
    cpdef double loss(self,p,y):
        cdef double z = p*y
        if z >= 1:
            return 0
        elif z >= -1:
            return (1-z) * (1-z) 
        else:
            return -4*z

    cpdef  double dloss(self,p,y):
        cdef double z = p*y
        if z >= 1:
            return 0
        elif z >= -1:
            return 2*(1-z)*y
        else:
            return 4*y

    def __reduce__(self):
        return ModifiedHuber,()

cdef class Hinge(Classification):
    """SVM classification loss for binary
    classification tasks with y in {-1,1}.
    """
    cpdef  double loss(self,p,y):
        cdef double z = p*y
        if z < 1:
            return (1 - z)
        return 0
    cpdef  double dloss(self,p,y):
        cdef double z = p*y
        if z < 1:
            return y
        return 0

    def __reduce__(self):
        return Hinge,()


cdef class Log(Classification):
    """Logistic regression loss for binary classification
    tasks with y in {-1,1}.
    """
    cpdef double loss(self,p,y):
        cdef double z = p*y
        if z > 18:
            return exp(-z)
        if z < -18:
            return -z * y
        return log(1.0+exp(-z)) 

    cpdef  double dloss(self,p,y):
        cdef double z = p*y
        if z > 18:
            return exp(-z) * y
        if z < -18:
            return y
        return y / (exp(z) + 1.0)

    def __reduce__(self):
        return Log,()

cdef class SquaredError(Regression):
    """
    """
    cpdef  double loss(self,p,y):
        return 0.5 * (p-y) * (p-y)
    cpdef  double dloss(self,p,y):
        return y - p

    def __reduce__(self):
        return SquaredError,()

cdef class Huber(Regression):
    """
    """
    cdef double c
    def __init__(self,c):
        self.c = c
    cpdef  double loss(self,p,y):
        cdef double r = p-y
        cdef double abs_r = abs(r)
        if abs_r <= self.c:
            return 0.5 * r * r
        else:
            return self.c * abs_r - (0.5*self.c*self.c)

    cpdef  double dloss(self,p,y):
        cdef double r = y - p 
        cdef double abs_r = abs(r)
        if abs_r <= self.c:
            return r
        elif r > 0:
            return self.c
        else:
            return -self.c

    def __reduce__(self):
        return Huber,(self.c,)

# ----------------------------------------
# Python function for external prediction
# ----------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def predict(np.ndarray x, np.ndarray w,
            double bias):
    cdef int xnnz = x.shape[0]
    cdef int wdim = w.shape[0]
    cdef double y = 0.0
    if xnnz == 0:
        y = bias
    else:
        y = dot_checked(<double *>w.data,<Pair *>x.data,xnnz,wdim) + bias
    return y
  
 # ----------------------------------------
 # C functions for fast sparse-dense vector operations
 # ----------------------------------------

cdef struct Pair:
    np.uint32_t idx
    np.float32_t val
    
cdef inline double max(double a, double b):
    return a if a >= b else b

cdef inline double min(double a, double b):
    return a if a <= b else b

cdef double dot(double *w, Pair *x, int nnz):
    """Dot product of weight vector w and example x. 
    """
    cdef double sum = 0.0
    cdef Pair pair
    for i from 0 <= i < nnz:
        pair = x[i]
        sum += w[pair.idx] * pair.val
    return sum

cdef double dot_checked(double *w, Pair *x, int nnz, int wdim):
    """ Checked version of dot product. Ignores features in x
    with a higher index than dimension of w. 
    """
    cdef double sum = 0.0
    cdef Pair pair
    for i from 0 <= i < nnz:
        pair = x[i]
        if pair.idx < wdim:
            sum +=w[pair.idx]*pair.val
    return sum

cdef void add(double *w, Pair *x, int nnz, double c):
    """Scales example x by constant c and adds it to the weight vector w. 
    """
    cdef Pair pair
    for i from 0 <= i < nnz:
        pair = x[i]
        w[pair.idx] += pair.val * c

# ----------------------------------------
# Extension type for Stochastic Gradient Descent
# ----------------------------------------

cdef class SGD:
    """Plain stochastic gradient descent solver.
    """
    cdef int epochs
    
    def __init__(self, epochs):
        self.epochs = epochs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def train(self, model, dataset, verbose = 0, shuffle = False):
        """Train `model` on the `dataset` using SGD.

        Parameters: 
        model: The LinearModel that is going to be trained. The model has to be configured properly. 
        dataset: The `Dataset`. 
        verbose: The verbosity level. If 0 no output to stdout.
        shuffle: Whether or not the training data should be shuffled after each epoch. 
        
        Engineering tricks:

          * explicit scaling factor of weight vector w. 

        References:

          * SGD implementation by Leon Buttuo.
          * PEGASOS algorithm by Shavel-Shwartz et al.
          * L1 penalty by Tsuruoka et al. 

        """
        cdef LossFunction loss = model.loss
        cdef int m = model.m
        cdef int n = dataset.n

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] w = model.w
        # weight vector w as c array
        cdef double *wdata = <double *>w.data
        # norm of w
        cdef double wscale = 1.0

        # training instance
        cdef np.ndarray x = None
        cdef Pair *xdata = NULL
        
        cdef double alpha = model.alpha
        cdef double b = 0.0,p,y,wnorm,t,update = 0.0
        cdef double reg = model.reg
        
        cdef int xnnz,nscale=0,nadd=0
        cdef int norm = model.norm
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] q = None
        cdef double *qdata
        cdef double u = 0.0
        cdef int usebias = 1
        cdef double sumloss = 0.0
        cdef int count = 0
        if model.biasterm == False:
            usebias = 0
        if norm == 1:
            q = np.zeros((m,), dtype = np.float64, order = "c" )
            qdata = <double *>q.data

        # computing eta0
        maxw = 1.0 / np.sqrt(reg)
        typw = np.sqrt(maxw)
        eta0 = typw /max(1.0,loss.dloss(-typw,1.0))
        t = 1.0 / (eta0 * reg)
        
        for e in range(self.epochs):
            if verbose > 0:
                print("-- Epoch %d" % (e+1))
            t1=time()
            nadd = nscale = 0
            if shuffle:
                dataset.shuffle()
            for x,y in dataset:
                eta = 1.0 / (reg * t)
                xnnz = x.shape[0]
                xdata = <Pair *>x.data
                p = (dot(wdata, xdata, xnnz) * wscale) + b
                sumloss += loss.loss(p,y)
                update = eta * loss.dloss(p,y)
                if update != 0:
                    add(wdata, xdata,
                        xnnz,(update / wscale))
                    if usebias == 1:
                        b += update * 0.01
                    nadd += 1

                if norm == 2:
                    wscale *= 1 - eta * reg
                    if wscale < 1e-9:
                        nscale += 1
                        w*=wscale
                        wscale = 1
                else:
                    u += reg*eta
                    l1penalty(wdata, qdata, xdata, xnnz, u)
                
                t += 1
                count += 1

            # report epoche information
            if verbose > 1:
                print("Scalings: %d, Adds: %d" %(nscale, nadd))
            if verbose > 0:
                wnorm = np.dot(w,w) * wscale * wscale
                print("Norm: %.2f, NNZs: %d, Bias: %.6f, T: %d, Avg. loss: %.6f" % (wnorm,w.nonzero()[0].shape[0],b,count,sumloss/count))
                print("Total training time: %.2f seconds." % (time()-t1))

        # floating-point under-/overflow check.
        if np.any(np.isinf(w)) or np.any(np.isnan(w))or np.isnan(b) or np.isinf(b):
            raise ValueError, "floating-point under-/overflow occured."
        model.w = w * wscale
        model.bias = b

cdef l1penalty(double *w, double *q, Pair *x, int nnz, double u):
    cdef double z = 0.0
    cdef Pair pair
    for i from 0 <= i < nnz:
        pair = x[i]
        j = pair.idx
        z = w[j]
        if w[j] > 0:
            w[j] = max(0,w[j] - (u + q[j]))
        elif w[j] < 0:
            w[j] = min(0,w[j] + (u - q[j]))
        q[j] += (w[j] - z)
