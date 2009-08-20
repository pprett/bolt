from __future__ import division

import sys
import numpy as np
from itertools import izip
from time import time
import cPickle as pickle

from bolt import version,predict,SGD,LossFunction,ModifiedHuber,Hinge,Log,SquaredError,Huber
from io import load, densedtype, sparsedtype, dense2sparse
import parse

loss_functions = {0:Hinge, 1:ModifiedHuber, 2:Log, 5:SquaredError, 6:Huber}

class LinearModel(object):
    """A linear model: y = x*w + b. 
    """

    def __init__(self, m, loss = ModifiedHuber(), reg = 0.001, alpha = 1.0):
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


    def __call__(self,x):
        """Predicts the target value for the given example. 
        
        Return:
        -------
        y = x*w + b
        """
        if x.dtype == sparsedtype:
            return predict(x, self.w, self.bias)
        else:
            sparsex = dense2sparse(x)
            return predict(sparsex, self.w, self.bias)

    def predict(self,examples):
        """Evaluates y = x*w + b for each
        example x in examples. 

        Parameters:
        examples: a sequence of examples. 
        """
        for x in examples:
            yield self.__call__(x)


def loadData(data_file, desc = "training", verbose = 1):
    if verbose > 0:
        print "loading %s data ..." % desc,
        
    sys.stdout.flush()
    try:
        examples, labels, dim = load(data_file)
    except IOError as (errno, strerror):
        if verbose > 0:
            print(" [fail]")
        raise Exception, "cannot open '%s' - %s." % (data_file,strerror)
    except Exception as exc:
        if verbose > 0:
            print(" [fail]")
        raise Exception, exc
    else:
        if verbose > 0:
            print(" [done]")

    if verbose > 1:
        print("%d (%d+%d) examples loaded. " % (len(examples),
                                                labels[labels==1.0].shape[0],
                                                labels[labels==-1.0].shape[0]))
    return examples, labels, dim

def errorrate(model,examples, labels):
    """Compute the misclassification rate of the model.

    Parameters:
    model: An instance of LinearModel

    examples: A sequence of sparse encoded examples.
    lables: A sequence of class labels, either 1 or -1. 
    """
    n = 0
    err = 0
    for p,y in izip(model.predict(examples),labels):
        z = p*y
        if np.isinf(p) or np.isnan(p) or z<0:
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

def computeError(lm,texamples,tlabels,loss):
    if type(loss) is not int:
        raise ValueError, "integer expected for loss."
    err = 100.0
    if loss < 5:
        err = errorrate(lm,texamples,tlabels)
    else:
        err = rmse(lm,texamples,tlabels)
    return err

def writePredictions(lm,texamples,tlabels,predictions_file):
    f = prediction_file
    out = sys.stdout if f == "-" else open(f,"w+")
    try:
        for p in lm.predict(examples):
            out.write("%.6f\n" % p)
    finally:
        out.close()
    

def main():
    try: 
        options, args, parser  = parse.parseArguments()
        if len(args) < 1 or len(args) > 3:
            parser.error("incorrect number of arguments. ")

        if options.test_only and not options.model_file:
            parser.error("option -m is required for --test-only.")

        if options.test_only and options.test_file:
            parser.error("options --test-only and -t are mutually exclusive.")

        verbose = options.verbose
        data_file = args[0]
        examples, labels, dim = loadData(data_file, verbose = verbose)
        
        if not options.test_only:
            if verbose > 0:
                print("---------")
                print("Training:")
                print("---------")
            loss_class = loss_functions[options.loss]
            loss = None
            if options.epsilon:
                loss = loss_class(options.epsilon)
            else:
                loss = loss_class()
            if not loss:
                raise Exception, "cannot create loss function."

            lm = LinearModel(dim,loss = loss, reg = options.regularizer, alpha = options.alpha)
            sgd = SGD(options.epochs)
            sgd.train(lm,examples,labels,verbose = verbose, shuffle = options.shuffle)
            err = computeError(lm,examples,labels, options.loss)
            if verbose > 0:
                print("error: %f%%." % (err))
            if options.model_file:
                f = open(options.model_file, 'w+')
                try:
                    pickle.dump(lm,f)
                finally:
                    f.close()
                
            if options.test_file:
                texamples, tlabels, tdim = loadData(options.test_file,
                                                    desc = "test", verbose = verbose)
                if options.prediction_file:
                    writePredictions(lm,texamples,tlabels,options.prediction_file)
                else:
                    print("--------")
                    print("Testing:")
                    print("--------")
                    t1 = time()
                    err = computeError(lm,texamples,tlabels,options.loss)
                    print("error: %f%%." % (err))
                    print("Total prediction time: %.2f seconds." % (time()-t1))
            
        else:
            lm = None
            f = open(options.model_file, 'r')
            try:
                lm = pickle.load(f)
            finally:
                f.close()
            if not lm:
                raise Exception, "cannot deserialize model in '%s'. " % options.model_file
            if options.prediction_file:
                writePredictions(lm,examples,labels,options.prediction_file)
            else:
                print("--------")
                print("Testing:")
                print("--------")
                t1 = time()
                err = computeError(lm,texamples,tlabels,options.loss)
                print("error: %f%%." % (err))
                print("Total prediction time: %.2f seconds." % (time()-t1))


    except Exception as exc:
        print "error: ", exc

if __name__ == "__main__":
    main() 


