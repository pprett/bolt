import copy
import numpy as np

import eval
import bolt
import parse

from time import time
from io import loadData
from model import LinearModel

version = "1.0"

def train(examples, labels, model,options):
    sgd = bolt.SGD(options.epochs)
    sgd.train(model,examples,labels,verbose = (options.verbose-1),
	      shuffle = options.shuffle)
    return model

def crossvalidation(examples, labels, model, options, seed = None):
    assert examples.shape[0] == labels.shape[0]
    verbose = options.verbose
    n = examples.shape[0]
    nfolds = options.nfolds
    r = n % nfolds
    num = n-r
    folds = np.array(np.split(examples[:num],nfolds))
    labels = np.array(np.split(labels[:num],nfolds))
    err = []
    for foldidx in range(nfolds):
	if verbose > 0:
	    print("--------------------")
	    print("Fold-%d" % (foldidx+1))
	    print("--------------------")
	lm = copy.deepcopy(model)
        t1 = time()
        test_examples = folds[foldidx]
        test_labels = labels[foldidx]
        trainidxs = range(nfolds)
        del trainidxs[foldidx]
        train_examples = np.concatenate(folds[trainidxs]) 
        train_labels = np.concatenate(labels[trainidxs])	
        lm = train(train_examples, train_labels, lm, options)
	e = eval.error(lm,test_examples,test_labels)
	if verbose > 0:
	    print("error: %.4f" % e)
        err.append(e)
	if options.verbose > 0:
	    print "Total time for fold-%d: %f" % (foldidx+1, time()-t1)
    return np.mean(err), np.std(err)
    

def main():
    try:
	parser  = parse.parseCV(version)
	options, args = parser.parse_args()
        if len(args) < 1 or len(args) > 1:
            parser.error("Incorrect number of arguments. ")

        
        verbose = options.verbose
        data_file = args[0]
        examples, labels, dim = loadData(data_file, verbose = verbose)

	loss_class = bolt.loss_functions[options.loss]
	loss = None
	if options.epsilon:
	    loss = loss_class(options.epsilon)
	else:
	    loss = loss_class()
	if not loss:
	    raise Exception, "Cannot create loss function."
	
        model = LinearModel(dim,loss = loss,
			    reg = options.regularizer,
			    alpha = options.alpha,
			    norm = options.norm)
	mean, std = crossvalidation(examples,labels, model, options)
	print "cv-error: %.4f (%.4f)" % (mean,std)

    except Exception, exc:
        print "[ERROR] ", exc


if __name__ == "__main__":
    main() 


