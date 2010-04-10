"""
Bolt
====

Bolt online learning toolbox.

Documentation is available in the docstrings. 

Subpackages
-----------

model
   Model specifications. 

bolt
   Extension module containing loss functions, and the SGD solver. 

io
   Input/Output routines; reading datasets, writing predictions.

eval
   Evaluation metrics. 

parse
   Command line parsing.

see http://github.com/pprett/bolt

"""
from __future__ import division

import sys
import numpy as np
import cPickle as pickle

from itertools import izip
from time import time

import parse
import eval

from trainer import OVA
from trainer.sgd import predict,SGD,LossFunction,Classification,Regression,Hinge, ModifiedHuber, Log, SquaredError, Huber, PEGASOS
from io import MemoryDataset,sparsedtype,dense2sparse
from model import LinearModel, GeneralizedLinearModel
from eval import errorrate

__version__ = "1.3"

loss_functions = {0:Hinge, 1:ModifiedHuber, 2:Log, 5:SquaredError, 6:Huber}

def writePredictions(lm,ds,pfile):
    """Write model predictions to file.
    The prediction file has as many lines as len(examples).
    The i-th line contains the prediction for the i-th example, encoded as
    a floating point number

    Parameters:
    lm: A `LinearModel`
    ds: A `Dataset`
    pfile: The filename to which predictions are written
    """
    f = pfile
    out = sys.stdout if f == "-" else open(f,"w+")
    try:
        for p in lm.predict(ds.iterinstances()):
            out.write("%.6f\n" % p)
    finally:
        out.close()
    
def main():
    try: 
        parser  = parse.parseSB(__version__)
	options, args = parser.parse_args()
        if len(args) < 1 or len(args) > 1:
            parser.error("incorrect number of arguments (use `--help` for help).")

        if options.test_only and not options.model_file:
            parser.error("option -m is required for --test-only.")

        if options.test_only and options.test_file:
            parser.error("options --test-only and -t are mutually exclusive.")

        verbose = options.verbose
        data_file = args[0]
        dtrain = MemoryDataset.load(data_file, verbose = verbose)
        
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
            
            lm = LinearModel(dtrain.dim,
			     biasterm = options.biasterm)

            if options.clstype == "sgd":
                trainer = SGD(loss, options.regularizer,
                          norm = options.norm,
                          epochs = options.epochs,
			  alpha = options.alpha)
            
            elif options.clstype == "pegasos":
                trainer = PEGASOS(options.regularizer,
                          epochs = options.epochs)
            else:
                parser.error("classifier type \"%s\" not supported." % options.clstype)
            trainer.train(lm,dtrain,verbose = verbose,
		      shuffle = options.shuffle)

	    if options.computetrainerror:
		err = eval.error(lm,dtrain,loss)
		print("error: %.4f" % err)
		sys.stdout.flush()
            if options.model_file:
                f = open(options.model_file, 'w+')
                try:
                    pickle.dump(lm,f)
                finally:
                    f.close()
                
            if options.test_file:
                dtest = MemoryDataset.load(options.test_file,
                                           verbose = verbose)
                
                if options.prediction_file:
                    writePredictions(lm,dtest,options.prediction_file)
                else:
                    print("--------")
                    print("Testing:")
                    print("--------")
                    t1 = time()
                    err = eval.error(lm,dtest,loss)
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
                writePredictions(lm,dtrain,options.prediction_file)
            else:
                print("--------")
                print("Testing:")
                print("--------")
                t1 = time()
                err = eval.errorrate(lm,dtrain)
		print("error: %.4f" % err)
                print("Total prediction time: %.2f seconds." % (time()-t1))

    except Exception, exc:
        print "[ERROR] ", exc

if __name__ == "__main__":
    main() 


