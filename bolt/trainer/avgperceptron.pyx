# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# filename: avgperceptron.pyx

"""
The :mod:`bolt.trainer.avgperceptron` module contains an implementation of the
averaged perceptron algorithm for multi-class classification. 

"""
from __future__ import division

import numpy as np
import sys

cimport numpy as np
cimport cython

from time import time

__authors__ = [
      '"Peter Prettenhofer" <peter.prettenhofer@gmail.com>'
]

cdef extern from "math.h":
    cdef extern double exp(double x)
    cdef extern double log(double x)
    cdef extern double sqrt(double x)

#cdef extern from "cblas.h":
#    double ddot "cblas_ddot"(int N,
#                             double *X, int incX,
#                             double *Y, int incY)
    

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

cdef void add(double *w, int stride, Pair *x, int nnz, int y, double c):
    """Scales example x by `c` and adds it to the weight vector w[k,:]. 
    """
    cdef Pair pair
    cdef int i
    cdef int offset = (y*stride)
    for i from 0 <= i < nnz:
        pair = x[i]
        w[offset + pair.idx] += (pair.val * c)

cdef void add_vector(double *wbar, double *w, double c, int length):
    """Adds w*c to wbar
    """
    cdef int i
    for i from 0 <= i < length:
        wbar[i] += c*w[i]

cdef int argmax(double *w, int wstride, Pair *x, int xnnz, int y, int k):
    cdef Pair pair
    cdef int j
    cdef double *wk
    cdef double max_score = -1.0
    cdef double p = 0.0
    cdef int max_j = 0
    for j from 0 <= j < k:
        wk = w + (wstride*j)
        p = dot(wk, x, xnnz)
        if p >= max_score:
            max_j = j
            max_score = p
    return max_j

cdef class AveragedPerceptron:
    """Averaged Perceptron learning algorithm. 

**References**:
   * [Yoav1999]_.
   * [Collins2002]_.
          
**Parameters**:
   * *epochs* - The number of iterations through the dataset. Default `epochs=5`. 
    """
    cdef int epochs
    
    def __init__(self, epochs = 5):
        """        
        :arg epochs: The number of iterations through the dataset.
        :type epochs: int
        """
        self.epochs = epochs
        

    def train(self, model, dataset, verbose = 0, shuffle = False):
        """Train `model` on the `dataset` using SGD.

        :arg model: The model that is going to be trained. Either :class:`bolt.model.GeneralizedLinearModel` or :class:`bolt.model.LinearModel`.
        :arg dataset: The :class:`bolt.io.Dataset`. 
        :arg verbose: The verbosity level. If 0 no output to stdout.
        :arg shuffle: Whether or not the training data should be shuffled after each epoch. 
        """
        self._train_multi(model, dataset, verbose, shuffle)

    cdef void _train_multi(self,model, dataset, verbose, shuffle):
        cdef int m = model.m
        cdef int k = model.k
        cdef int n = dataset.n
        cdef int length = k*m

        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] w = model.W
        # maintain a averaged w
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] wbar = np.zeros((k,m),
                                                                        dtype = np.float64)
        cdef double *wdata = <double *>w.data
        cdef double *wbardata = <double *>wbar.data
        cdef int wstride0 = w.strides[0]
        cdef int wstride1 = w.strides[1]
        cdef int wstride = wstride0 / wstride1

        # training instance
        cdef np.ndarray x = None
        cdef Pair *xdata = NULL
        cdef float y = 0
        cdef int yhat = 0
        cdef int xnnz = 0, t = 0, nadds = 0
        cdef int u = 1
        t1=time()
        for e from 0 <= e < self.epochs:
            if verbose > 0:
                print("-- Epoch %d" % (e+1))
            if shuffle:
                dataset.shuffle()
            nadds = 0
            for x,y in dataset:
                xnnz = x.shape[0]
                xdata = <Pair *>x.data
                yhat = argmax(wdata, wstride, xdata, xnnz, <int>y, k)
                if yhat != y:
                    add_vector(wbardata, wdata, u, length)
                    u = 0
                    add(wdata, wstride, xdata, xnnz, <int>y, 1)
                    add(wdata, wstride, xdata, xnnz, yhat, -1)
                    nadds += 1

                u += 1
                t += 1
            # report epoche information
            if verbose > 0:
                print("NADDs: %d; NNZs: %d. " % (nadds, w.nonzero()[0].shape[0]))
                print("Total training time: %.2f seconds." % (time()-t1))
        
        add_vector(wbardata, wdata, u, length)
        # floating-point under-/overflow check.
        if np.any(np.isinf(w)) or np.any(np.isnan(w)):
            raise ValueError, "floating-point under-/overflow occured."
        
        model.W = wbar * (1.0 / (n*self.epochs))
        
