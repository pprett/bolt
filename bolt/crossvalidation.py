import copy
import numpy as np

import eval
import bolt
import parse

from time import time
from io import MemoryDataset
from model import LinearModel

version = "1.1"

def train(ds, model,options):
    sgd = bolt.SGD(options.epochs)
    sgd.train(model,ds,verbose = (options.verbose-1),
	      shuffle = options.shuffle)
    return model

def crossvalidation(ds, model, options, seed = None):
    verbose = options.verbose
    n = ds.n
    nfolds = options.nfolds
    folds = ds.split(nfolds)
    err = []
    for foldidx in range(nfolds):
	if verbose > 1:
	    print("--------------------")
	    print("Fold-%d" % (foldidx+1))
	    print("--------------------")
	lm = copy.deepcopy(model)
        t1 = time()
        dtest = folds[foldidx]
        trainidxs = range(nfolds)
        del trainidxs[foldidx]
        dtrain = MemoryDataset.merge(folds[trainidxs])
        lm = train(dtrain, lm, options)
	e = eval.error(lm,dtest)
	if verbose > 0:
	    fid = ("fold-%d" % (foldidx+1)).ljust(8)
	    print("error %s %.4f" % (fid , e))
        err.append(e)
	if verbose > 1:
	    print "Total time for fold-%d: %f" % (foldidx+1, time()-t1)
    return np.mean(err), np.std(err)
    
def main():
    try:
	parser  = parse.parseCV(version)
	options, args = parser.parse_args()
        if len(args) < 1 or len(args) > 1:
            parser.error("Incorrect number of arguments. ")
        
        verbose = options.verbose
        fname = args[0]
        ds = MemoryDataset.load(fname,verbose = verbose)

	loss_class = bolt.loss_functions[options.loss]
	loss = None
	if options.epsilon:
	    loss = loss_class(options.epsilon)
	else:
	    loss = loss_class()
	if not loss:
	    raise Exception, "Cannot create loss function."
	
        model = LinearModel(ds.dim,loss = loss,
			    reg = options.regularizer,
			    alpha = options.alpha,
			    norm = options.norm)
	mean, std = crossvalidation(ds, model, options)
	print("%s %s %.4f (%.4f)" % ("error","avg".ljust(8), mean,std))

    except Exception, exc:
        print "[ERROR] ", exc


if __name__ == "__main__":
    main() 


