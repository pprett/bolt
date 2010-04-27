# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# filename: maxent.pyx
from __future__ import division

import numpy as np
import sys

cimport numpy as np
cimport cython

from time import time
from itertools import izip

__authors__ = [
      '"Peter Prettenhofer" <peter.prettenhofer@gmail.com>'
]

cdef extern from "math.h":
    cdef extern double exp(double x)
    cdef extern double log(double x)
    cdef extern double sqrt(double x)
    cdef extern double pow(double base, double exponent)


 # ----------------------------------------
 # C functions for fast sparse-dense vector operations
 # ----------------------------------------

cdef struct Pair:
    np.uint32_t idx
    np.float32_t val

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

cdef double add(double *w, int stride, double wscale, double *bdata,
                Pair *x, int nnz, int y, double c):
    """Scales example x by constant c and adds it to the weight vector w. 
    """
    cdef Pair pair
    cdef int i
    cdef double innerprod = 0.0
    cdef double xsqnorm = 0.0
    cdef int offset = y * stride
    for i from 0 <= i < nnz:
        pair = x[i]
        w[offset + pair.idx] += pair.val * (c / wscale)
    bdata[y] += (c * 0.01) # update bias 
    

cdef double probdist(double *w, int wstride, double wscale, double *b,
                    Pair *x, int xnnz, int k, double *pd):
     cdef Pair pair
     cdef int j
     cdef double *wk
     cdef double sum = 0.0
     for j from 0 <= j < k:
         wk = w + (wstride*j)
         pd[j] = exp(dot(wk, x, xnnz) * wscale + b[j])
         sum += pd[j]
     for j from 0 <= j < k:
         pd[j] /= sum
     return sum


# ----------------------------------------
# Extension type for Stochastic Gradient Descent
# ----------------------------------------

cdef class MaxentSGD:
    """Stochastic gradient descent solver for maxent (aka multinomial logistic regression). The solver supports various penalties (L1, L2, and Elastic-Net). 

**References**:
   * SGD implementation inspired by Leon Buttuo's sgd and [Zhang2004]_.
   * L1 penalty via truncation [Tsuruoka2009]_.
   * Elastic-net penalty [Zou2005]_.
          
**Parameters**:
   * *reg* -  The regularization parameter lambda.
   * *epochs* - The number of iterations through the dataset. Default `epochs=5`. 
   * *norm* - Whether to minimize the L1, L2 norm or the Elastic Net (either 1,2, or 3; default 2).
   * *alpha* - The elastic net penality parameter (0<=`alpha`<=1). A value of 1 amounts to L2 regularization whereas a value of 0 gives L1 penalty (requires `norm=3`). Default `alpha=0.85`.
    """
    cdef int epochs
    cdef double reg
    cdef int norm
    
    def __init__(self, reg, epochs = 5, norm = 2):
        """
        :arg reg: The regularization parameter lambda (>0).
        :type reg: float.
        :arg epochs: The number of iterations through the dataset.
        :type epochs: int
        :arg norm: Whether to minimize the L1, L2 norm or the Elastic Net.
        :type norm: 1 or 2 or 3
        :arg alpha: The elastic net penality parameter. A value of 1 amounts to L2 regularization whereas a value of 0 gives L1 penalty. 
        :type alpha: float (0 <= alpha <= 1)
        """
        if reg < 0.0:
            raise ValueError, "reg must be larger than 0. "
        if norm not in [1,2,3]:
            raise ValueError, "norm must be in {1,2,3}. "
        self.reg = reg
        self.epochs = epochs
        self.norm = norm
        

    def train(self, model, dataset, verbose = 0, shuffle = False):
        """Train `model` on the `dataset` using SGD.

        :arg model: The :class:`bolt.model.GeneralizedLinearModel` that is going to be trained. 
        :arg dataset: The :class:`bolt.io.Dataset`. 
        :arg verbose: The verbosity level. If 0 no output to stdout.
        :arg shuffle: Whether or not the training data should be shuffled after each epoch. 
        """
        self._train(model, dataset, verbose, shuffle)

    cdef void _train(self,model, dataset, verbose, shuffle):
        
        cdef int m = model.m
        cdef int n = dataset.n
        cdef int k = model.k
        cdef double reg = self.reg
        cdef int length = k*m

        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] w = model.W
        cdef int wstride0 = w.strides[0]
        cdef int wstride1 = w.strides[1]
        cdef int wstride = wstride0 / wstride1
        # weight vector w as c array
        cdef double *wdata = <double *>w.data
        # the scale of w
        cdef double wscale = 1.0

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] b = model.b
        cdef double *bdata = <double *>b.data

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] PD = np.zeros((k,),
                                                                            dtype = np.float64)
        cdef double *pd = <double *>PD.data

        # training instance
        cdef np.ndarray x = None
        cdef Pair *xdata = NULL
        cdef float y = 0.0
        cdef int z = -1
        
        # bias term (aka offset or intercept)
        cdef int usebias = 1
        if model.biasterm == False:
            usebias = 0

        cdef double wnorm = 0.0, update = 0.0,sumloss = 0.0, eta = 0.0, t = 0.0
        cdef int xnnz = 0, i = 0, e = 0, j = 0
        
        cdef double typw = sqrt(1.0 / sqrt(reg))
        cdef double eta0 = typw / 1.0
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
                probdist(wdata, wstride, wscale, bdata, xdata, xnnz, k, pd)
                add(wdata, wstride, wscale, bdata, xdata, xnnz, <int>y, eta)
                for j from 0 <= j < k:
                    add(wdata, wstride, wscale, bdata,
                        xdata, xnnz, j, -1.0 * eta * pd[j])
                wscale *= (1 - eta * (reg))
                if wscale < 1e-9:
                    w*=wscale
                    wscale = 1.0
                t += 1

            # report epoche information
            if verbose > 0:
                print("Wscale: %.6f" % (wscale))
                print("Total training time: %.2f seconds." % (time() - t1))

        # floating-point under-/overflow check.
        if np.any(np.isinf(w)) or np.any(np.isnan(w)):
            raise ValueError, "floating-point under-/overflow occured."
        model.w = w * wscale
        model.bias = b
