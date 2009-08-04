from __future__ import division

import sys
import numpy as np
from itertools import izip
from time import time

from bolt import version,predict,SGD,LossFunction,ModifiedHuber,Hinge,Log
from io import load
import parse

def error(model,examples, labels):
    """Compute the missclassification rate of the model.

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

class LinearModel(object):
    """A linear model: y = x*w + b. 
    """

    def __init__(self, m):
        """Create a linear model with an
        m-dimensional vector w = [0,..,0] and b = 0.

        Parameters:
        m: The dimensionality of the classification problem (i.e. the number of features). 
        """
        self.m = m
        self.w = np.zeros((m),dtype=np.float64)
        self.bias = 0.0

    def __call__(self,x):
        """Predicts the target value for the given example. 
        TODO: implement sparse and dense classification.
        
        Return:
        -------
        y = x*w + b
        """
        return predict(x, self.w, self.bias)
        

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

    if verbose > 0:
        print("%d (%d+%d) examples loaded. " % (len(examples),
                                                labels.count(1),
                                                labels.count(-1)))
    return examples, labels, dim
    

def main():
    try: 
        options, args, parser  = parse.parseArguments()
        if len(args) < 1 or len(args) > 3:
            parser.error("incorrect number of arguments. ")
            
        verbose = options.verbose
        data_file = args[0]
        examples, labels, dim = loadData(data_file, verbose = verbose)
        lm = LinearModel(dim)
        if not options.testonly:
            sgd = SGD(options.epochs, options.regularizer)
            if verbose > 0:
                print("---------")
                print("Training:")
                print("---------\n")
            sgd.train(lm,options.loss,examples,labels,verbose = verbose, shuffle = options.shuffle)
        else:
            raise Exception, "implement load model and test only"

        if options.prediction_file:
            f = options.prediction_file
            out = sys.stdout if f == "-" else open(f,"w+")
            try:
                for p in lm.predict(examples):
                    out.write("%.6f\n" % p)
            finally:
                out.close()
        else:                
            errrate = error(lm,examples,labels)
            if verbose > 0:
                print("error rate: %f%%." % (errrate))

        if options.test_file:
            texamples, tlabels, tdim = loadData(options.test_file,
                                             desc = "test", verbose = verbose)
            #if tdim > dim:
            #    raise Exception, "Dimensionality of test data is larger than training data. "
            print("\n")
            print("--------")
            print("Testing:")
            print("--------\n")
            t1 = time()
            errrate = error(lm,texamples,tlabels)
            print("error rate: %f%%." % (errrate))
            print("Total prediction time: %.2f seconds." % (time()-t1))

    except Exception as exc:
        print "error: ", exc

if __name__ == "__main__":
    main()


