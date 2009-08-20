from __future__ import division

import numpy as np
import sys

cimport numpy as np

from time import time
from itertools import izip

version = "0.1"

cdef extern from "math.h":
    cdef extern double exp(double x)
    cdef extern double log(double x)

# ----------------------------------------
# Extension Types for Loss Functions
# ----------------------------------------

cdef class LossFunction:
    """Base class for convex loss functions"""
    cpdef double loss(self,p, y):
        raise NotImplementedError()
    cpdef double dloss(self,p, y):
        raise NotImplementedError()

cdef class ModifiedHuber(LossFunction):
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
            return (1-z) * (1-z) * y
        else:
            return -4*z*y

    cpdef  double dloss(self,p,y):
        cdef double z = p*y
        if z >= 1:
            return 0
        elif z >= -1:
            return 2*(1-z)*y
        else:
            return 4*y

cdef class Hinge(LossFunction):
    """SVM classification loss for binary
    classification tasks with y in {-1,1}.
    """
    cpdef  double loss(self,p,y):
        cdef double z = p*y
        if z < 1:
            return (1 - z) * y
        return 0
    cpdef  double dloss(self,p,y):
        cdef double z = p*y
        if z < 1:
            return 1*y
        return 0

cdef class Log(LossFunction):
    """Logistic regression loss for binary classification
    tasks with y in {-1,1}.
    """
    cpdef double loss(self,p,y):
        cdef double z = p*y
        if z > 18:
            return exp(-z) * y
        if z < -18:
            return -z * y
        return log(1.0+exp(-z)) * y

    cpdef  double dloss(self,p,y):
        cdef double z = p*y
        if z > 18:
            return exp(-z) * y
        if z < -18:
            return y
        return y / (exp(z) + 1.0)

cdef class SquaredError(LossFunction):
    """
    """
    cpdef  double loss(self,p,y):
        return 0.5 * (p-y) * (p-y)
    cpdef  double dloss(self,p,y):
        return p-y

cdef class Huber(LossFunction):
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
        cdef double r = p-y
        cdef double abs_r = abs(r)
        if abs_r <= self.c:
            return r
        elif r > 0:
            return self.c
        else:
            return -self.c

# ----------------------------------------
# Python function for external prediction
# ----------------------------------------

def predict(example, np.ndarray w, double bias):
    cdef object[Pair] x = example
    cdef int xnnz = x.shape[0]
    cdef int wdim = w.shape[0]
    cdef double y = 0.0
    if xnnz == 0:
        y = bias
    else:
        y = dot_checked(<double *>w.data,<Pair *>&(x[0]),xnnz,wdim) + bias
    return y
  
 # ----------------------------------------
 # C functions for fast sparse-dense vector operations
 # ----------------------------------------

cdef struct Pair:
    np.uint32_t idx
    np.float32_t val
    
cdef inline double max(double a, double b):
    return a if a >= b else b

cdef double dot(double *w, Pair *x, int nnz):
    """Dot product of weight vector w and example x. 
    """
    cdef double sum = 0.0
    cdef Pair pair
    for i from 0 <= i < nnz:
        pair = x[i]
        sum +=w[pair.idx]*pair.val
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

    def train(self, model, examples, labels, verbose = 0, shuffle = False):
        """

        Parameters: 

        Structure:

        Engineering tricks:

          * explicit scaling factor of weight vector w. 

        References:

          * SGD implementation by Leon Buttuo. 

        """
        cdef LossFunction loss = model.loss
        cdef int m = model.m
        cdef int n = len(examples)
        cdef np.ndarray w = model.w
        cdef double *wdata = <double *>w.data
        cdef double wscale = 1.0
        cdef double alpha = model.alpha
        cdef double bias = 0.0,z,p,t,y,wnorm, s = 0.0
        cdef double reg = model.reg
        cdef object[Pair] x = None
        cdef Pair *xdata = NULL
        cdef int xnnz,nscale,nadd,count
        cdef double maxw = 1.0 / np.sqrt(reg)
        cdef double typw = np.sqrt(maxw)
        cdef double eta0 = typw /max(
            1.0,loss.dloss(-typw,1.0))
        t = 1.0 / (eta0 * reg)
        #print "maxw: %f, typw: %f, eta0: %f, t: %f" % (maxw,typw,eta0,t)
        
        for e in range(self.epochs):
            if verbose > 0:
                print("-- Epoch %d" % (e+1))
            t1=time()
            nadd = nscale = 0
            if shuffle:
                data = zip(examples,labels)
                np.random.shuffle(data)
                examples,labels = zip(*data)
            count = 0
            for x,y in izip(examples,labels):
                eta = 1.0 / (reg * t)
                s = 1 - eta * reg * alpha
                wscale *= s
                if alpha < 1.0:
                    if count == 0:
                        mask = np.sign(w) # np.greater(w*w,0.0)
                        count = 10
                    w -= (eta * reg * (1.0 - alpha)  / wscale) * mask
                    count -= 1
                if wscale < 1e-9:
                    nscale += 1
                    w*=wscale
                    wscale = 1
                xnnz = x.shape[0]
                if xnnz == 0: # handle all zero input examples. 
                    p = bias
                else:
                    xdata = <Pair *>&(x[0])
                    p = (dot(wdata, xdata,
                         xnnz) * wscale) + bias
                etd = eta * loss.dloss(p,y)
                if etd != 0:
                    add(wdata, xdata,
                        xnnz,(etd / wscale))
                    bias += etd * 0.01
                    nadd += 1
                t += 1
                                
            # floating-point under-/overflow check.
            if np.any(np.isinf(w)) or np.any(np.isnan(w)) or np.isnan(bias) or np.isinf(bias):
                #print bias, np.any(np.isinf(w)), np.any(np.isnan(w))
                raise ValueError, "floating-point under-/overflow occured."

            # report epoche information
            wnorm = np.dot(w,w) * wscale * wscale
            if verbose > 1:
                print("Scalings: %d, Adds: %d" %(nscale, nadd))
            if verbose > 0:
                print("Norm: %.2f, NNZs: %d, Bias: %.6f" % (wnorm,w.nonzero()[0].shape[0],bias))
                print("Total training time: %.2f seconds." % (time()-t1))

        model.w = w * wscale
        model.bias = bias

        
