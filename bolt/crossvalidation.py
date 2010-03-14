import copy
import numpy as np

import eval
import bolt
import parse

from time import time
from io import MemoryDataset
from model import LinearModel

version = "1.1"

def crossvalidation(ds, trainer, model, nfolds = 10, verbose = 1, shuffle = False, error = eval.errorrate, seed = None):
    n = ds.n
    ds.shuffle(seed = seed)
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
        trainer.train(lm, dtrain,
                      verbose = (verbose-1),
                      shuffle = shuffle)
	e = error(lm,dtest)
	if verbose > 0:
	    fid = ("%d" % (foldidx+1)).ljust(5)
	    print("%s %.4f" % (fid , e))
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

        lm = LinearModel(ds.dim,
			 biasterm = options.biasterm)
        if options.clstype == "sgd":
            trainer = bolt.SGD(loss, options.regularizer,
                          norm = options.norm,
                          alpha = options.alpha,
                          epochs = options.epochs)
            
        elif options.clstype == "pegasos":
            trainer = bolt.PEGASOS(options.regularizer,
                          epochs = options.epochs)
        else:
            parser.error("classifier type \"%s\" not supported." % options.clstype)
        print("%s %s" % ("Fold".ljust(5), "Error"))
	mean, std = crossvalidation(ds, trainer, lm,
                                    nfolds = options.nfolds,
                                    shuffle = options.shuffle,
                                    error = eval.errorrate,
                                    verbose = options.verbose,
                                    seed = options.seed)
	print("%s %.4f (%.4f)" % ("avg".ljust(5), mean,std))

    except Exception, exc:
        print "[ERROR] ", exc


if __name__ == "__main__":
    main() 


