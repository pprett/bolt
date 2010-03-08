from __future__ import division

import sys
import numpy as np
import cPickle as pickle

from itertools import izip
from time import time

import parse
import eval

from bolt import predict,SGD,LossFunction,Classification,Regression,loss_functions, Hinge, ModifiedHuber, Log, SquaredError, Huber
from io import loadData,sparsedtype,dense2sparse
from model import LinearModel
from eval import errorrate

version = "1.1"

def writePredictions(lm,examples,predictions_file):
    """Write model predictions to file.
    The prediction file has as many lines as len(examples).
    The i-th line contains the prediction for the i-th example, encoded as
    a floating point number """
    f = predictions_file
    out = sys.stdout if f == "-" else open(f,"w+")
    try:
        for p in lm.predict(examples):
            out.write("%.6f\n" % p)
    finally:
        out.close()
    

def main():
    try: 
        parser  = parse.parseSB(version)
	options, args = parser.parse_args()
        if len(args) < 1 or len(args) > 1:
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

            lm = LinearModel(dim,loss = loss,
			     reg = options.regularizer,
			     alpha = options.alpha,
			     norm = options.norm,
			     biasterm = options.biasterm)
            sgd = SGD(options.epochs)
            sgd.train(lm,examples,labels,verbose = verbose,
		      shuffle = options.shuffle)
            err = eval.error(lm,examples,labels)
	    print("error: %.4f" % err)
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
                    writePredictions(lm,texamples,options.prediction_file)
                else:
                    print("--------")
                    print("Testing:")
                    print("--------")
                    t1 = time()
                    err = eval.error(lm,texamples,tlabels)
		    print("error: %.4f" % err)
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
                writePredictions(lm,examples,options.prediction_file)
            else:
                print("--------")
                print("Testing:")
                print("--------")
                t1 = time()
                err = eval.error(lm,examples,labels)
		print("error: %.4f" % err)
                print("Total prediction time: %.2f seconds." % (time()-t1))

    except Exception, exc:
        print "[ERROR] ", exc

if __name__ == "__main__":
    main() 


