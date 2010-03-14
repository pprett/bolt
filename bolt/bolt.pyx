# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
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

cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int N,
                             double *X, int incX,
                             double *Y, int incY)
    
# ----------------------------------------
# Extension Types for Loss Functions
# ----------------------------------------

cdef class LossFunction:
    """Base class for convex loss functions"""
    cpdef double loss(self,double p,double y):
        raise NotImplementedError()
    cpdef double dloss(self,double p,double y):
        raise NotImplementedError()

cdef class Regression(LossFunction):
    """Base class for loss functions for regression."""
    cpdef double loss(self,double p,double y):
        raise NotImplementedError()
    cpdef double dloss(self,double p,double y):
        raise NotImplementedError()


cdef class Classification(LossFunction):
    """Base class for loss functions for classification."""
    cpdef double loss(self,double p,double y):
        raise NotImplementedError()
    cpdef double dloss(self,double p,double y):
        raise NotImplementedError()

cdef class ModifiedHuber(Classification):
    """Modified Huber loss function for binary
    classification tasks with y in {-1,1}.
    Its equivalent to quadratically smoothed SVM
    with gamma = 2. See T. Zhang 'Solving
    Large Scale Linear Prediction Problems Using
    Stochastic Gradient Descent', ICML'04.
    """
    cpdef double loss(self,double p,double y):
        cdef double z = p*y
        if z >= 1:
            return 0
        elif z >= -1:
            return (1-z) * (1-z) 
        else:
            return -4*z

    cpdef double dloss(self,double p,double y):
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
    cpdef  double loss(self,double p,double y):
        cdef double z = p*y
        if z < 1.0:
            return (1 - z)
        return 0
    cpdef  double dloss(self,double p,double y):
        cdef double z = p*y
        if z < 1.0:
            return y
        return 0

    def __reduce__(self):
        return Hinge,()


cdef class Log(Classification):
    """Logistic regression loss for binary classification
    tasks with y in {-1,1}.
    """
    cpdef double loss(self,double p,double y):
        cdef double z = p*y
        if z > 18:
            return exp(-z)
        if z < -18:
            return -z * y
        return log(1.0+exp(-z)) 

    cpdef  double dloss(self,double p,double y):
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
    cpdef  double loss(self,double p,double y):
        return 0.5 * (p-y) * (p-y)
    cpdef  double dloss(self,double p,double y):
        return y - p

    def __reduce__(self):
        return SquaredError,()

cdef class Huber(Regression):
    """
    """
    cdef double c
    def __init__(self,c):
        self.c = c
    cpdef  double loss(self,double p,double y):
        cdef double r = p-y
        cdef double abs_r = abs(r)
        if abs_r <= self.c:
            return 0.5 * r * r
        else:
            return self.c * abs_r - (0.5*self.c*self.c)

    cpdef  double dloss(self,double p,double y):
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
    cdef int i
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
    cdef int i
    for i from 0 <= i < nnz:
        pair = x[i]
        if pair.idx < wdim:
            sum +=w[pair.idx]*pair.val
    return sum

cdef double add(double *w, double wscale, Pair *x, int nnz, double c):
    """Scales example x by constant c and adds it to the weight vector w. 
    """
    cdef Pair pair
    cdef int i
    cdef double innerprod = 0.0
    cdef double xsqnorm = 0.0
    for i from 0 <= i < nnz:
        pair = x[i]
        innerprod += (w[pair.idx] * pair.val)
        xsqnorm += (pair.val*pair.val)
        w[pair.idx] += pair.val * (c / wscale)
        
    return (xsqnorm * c * c) + (2.0 * innerprod * wscale * c)

# ----------------------------------------
# Extension type for Stochastic Gradient Descent
# ----------------------------------------

cdef class SGD:
    """Plain stochastic gradient descent solver.
    """
    cdef int epochs
    cdef double reg
    cdef LossFunction loss
    cdef int norm
    cdef double alpha
    
    def __init__(self, loss, reg, epochs = 5, norm = 2, alpha = 1.0):
        """
        Parameters:
        loss: The loss function (default ModifiedHuber) 
        reg: The regularization parameter lambda.
        alpha: The elastic net hyper-paramter alpha. Blends L2 and L1 norm regularization (default 1.0).
        """
        if loss == None:
            raise ValueError, "Loss function must not be None."
        if reg < 0.0:
            raise ValueError, "reg must be larger than 0. "
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError, "alpha must be in [0,1]."
        self.loss = loss
        self.reg = reg
        self.epochs = epochs
        self.norm = norm
        self.alpha = alpha

    def train(self, model, dataset, verbose = 0, shuffle = False):
        """Train `model` on the `dataset` using SGD.

        Parameters: 
        model: The LinearModel that is going to be trained. The model has to be configured properly. 
        dataset: The `Dataset`. 
        verbose: The verbosity level. If 0 no output to stdout.
        shuffle: Whether or not the training data should be shuffled after each epoch. 
        
        References:

          * SGD implementation by Leon Buttuo.
          * L1 penalty by Tsuruoka et al. 

        """
        self._train(model, dataset, verbose, shuffle)

    cdef void _train(self,model, dataset, verbose, shuffle):
        
        cdef LossFunction loss = self.loss
        cdef int m = model.m
        cdef int n = dataset.n
        cdef double reg = self.reg

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] w = model.w
        # weight vector w as c array
        cdef double *wdata = <double *>w.data
        # the scale of w
        cdef double wscale = 1.0

        # training instance
        cdef np.ndarray x = None
        cdef Pair *xdata = NULL

        cdef double y = 0.0
        
        # Variables for penalty term
        cdef int norm = self.norm
        cdef double alpha = self.alpha
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] q = None
        cdef double *qdata = NULL
        cdef double u = 0.0
        if norm == 1:
            q = np.zeros((m,), dtype = np.float64, order = "c" )
            qdata = <double *>q.data

        # bias term (aka offset or intercept)
        cdef int usebias = 1
        if model.biasterm == False:
            usebias = 0

        cdef double b = 0.0,p=0.0,wnorm=0.0,t=0.0,update = 0.0,sumloss = 0.0,eta=0.0
        cdef int xnnz = 0, count = 0, i=0
        
        # computing eta0
        maxw = 1.0 / np.sqrt(reg)
        typw = np.sqrt(maxw)
        eta0 = typw /max(1.0,loss.dloss(-typw,1.0))
        t = 1.0 / (eta0 * reg)
        
        t1=time()
        for e from 0 <= e < self.epochs:
            if verbose > 0:
                print("-- Epoch %d" % (e+1))
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
                    add(wdata, wscale, xdata,
                        xnnz,update)
                    if usebias == 1:
                        b += update * 0.01

                if norm == 2:
                    wscale *= 1 - eta * reg
                    if wscale < 1e-9:
                        w*=wscale
                        wscale = 1
                else:
                    u += reg*eta
                    l1penalty(wdata, qdata, xdata, xnnz, u)
                
                t += 1
                count += 1

            # report epoche information
            if verbose > 0:
                wnorm = sqrt(np.dot(w,w) * wscale * wscale)
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
    cdef int i,j
    for i from 0 <= i < nnz:
        pair = x[i]
        j = pair.idx
        z = w[j]
        if w[j] > 0:
            w[j] = max(0,w[j] - (u + q[j]))
        elif w[j] < 0:
            w[j] = min(0,w[j] + (u - q[j]))
        q[j] += (w[j] - z)

########################################
#
# PEGASOS
#
########################################
        
cdef class PEGASOS:
    """PEGASOS SVM solver.

    [Shwartz, S. S., Singer, Y., and Srebro, N., 2007] Pegasos: Primal
    estimated sub-gradient solver for svm. In ICML '07: Proceedings of the
    24th international conference on Machine learning, pages 807-814, New
    York, NY, USA. ACM.
    """
    cdef int epochs
    cdef double reg
    
    def __init__(self, reg, epochs):
        if reg < 0.0:
            raise ValueError, "`reg` must be larger than 0. "
        self.epochs = epochs
        self.reg = reg

    def train(self, model, dataset, verbose = 0, shuffle = False):
        self._train(model, dataset, verbose, shuffle)

    cdef void _train(self, model, dataset, verbose, shuffle):
        """Train `model` on the `dataset` using PEGASOS.

        Parameters: 
        model: The `LinearModel` that is going to be trained. 
        dataset: The `Dataset`. 
        verbose: The verbosity level. If 0 no output to stdout.
        shuffle: Whether or not the training data should be shuffled after each epoch. 
        """
        cdef int m = model.m
        cdef int n = dataset.n
        cdef double reg = self.reg
        cdef double invsqrtreg = 1.0 / np.sqrt(reg)

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] w = model.w
        # weight vector w as c array
        cdef double *wdata = <double *>w.data
        # norm of w
        cdef double wscale = 1.0
        cdef double wnorm = 0.0
        
        # training instance
        cdef np.ndarray x = None
        cdef Pair *xdata = NULL
        
        cdef double b = 0.0,p = 0.0,update = 0.0, z = 0.0
        cdef double y = 0.0
        cdef double eta = 0.0
        
        cdef int xnnz=0
        
        cdef int usebias = 1
        cdef double sumloss = 0.0
        cdef int t = 1, i = 0

        t1=time()
        for e from 0 <= e < self.epochs:
            if verbose > 0:
                print("-- Epoch %d" % (e+1))
            
            if shuffle:
                dataset.shuffle()
            for x,y in dataset:
                eta = 1.0 / (reg * t)
                xnnz = x.shape[0]
                xdata = <Pair *>x.data
                p = (dot(wdata, xdata, xnnz) * wscale) + b
                z = p*y
                if z < 1:
                    wnorm += add(wdata, wscale, xdata,
                                 xnnz,(eta*y))
                    sumloss += (1-z)
                scale(&wscale, &wnorm, 1 - (eta * reg))
                project(wdata, &wscale, &wnorm, reg)
                if wscale < 1e-11:
                    w *= wscale
                    wscale = 1.0
                    
                t += 1

            if verbose > 0:
                print("Norm: %.2f, NNZs: %d, Bias: %.6f, T: %d, Avg. loss: %.6f" % (sqrt(wnorm),w.nonzero()[0].shape[0],b,t+1,sumloss/(t+1)))
                print("Total training time: %.2f seconds." % (time()-t1))
                
        # floating-point under-/overflow check.
        if np.any(np.isinf(w)) or np.any(np.isnan(w))or np.isnan(b) or np.isinf(b):
            raise ValueError, "floating-point under-/overflow occured."
        model.w = w * wscale
        model.bias = b

cdef inline void project(double *wdata, double *wscale, double *wnorm, double reg):
    """Project w onto L2 ball.
    """
    cdef double val = 1.0 
    if (wnorm[0]) != 0:
        val = 1.0 / sqrt(reg *  wnorm[0])
        if val < 1.0:
            scale(wscale,wnorm,val)    

cdef inline void scale(double *wscale, double *wnorm, double factor):
    """Scale w by constant factor. Update wnorm too.
    """
    wscale[0] *= factor
    wnorm[0] *= (factor*factor)
