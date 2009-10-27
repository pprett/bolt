import sys
import numpy as np
import gzip
#from time import time

"""The data type of the 
"""
sparsedtype = np.dtype("u4,f4")
densedtype = np.float32

def dense2sparse(x):
    return np.array([(nnz, x[nnz]) for nnz in x.nonzero()[0]],dtype = sparsedtype)
    
def load(filename):
    """Load a dataset from the given filename.

    If the filename ends with '.npy' it is assumed to be stored in binary format as a numpy archive (first example array, second labels; finally number of dimensions).

    Else it is assumed to be stored in svm^light format (textual). Number of dimensions are computed on the fly. 
    """
    if filename.endswith('.npy'):
        return loadNpz(filename)
    else:
        return loadDat(filename)

def loadNpz(filename):
    f = open(filename,'r')
    try:
        examples = np.load(f)
        labels = np.load(f)
        dim = np.load(f)
        return examples,labels,dim
    finally:
        f.close()

def loadDat(filename):
    labels = []
    examples = []
    qids = []
    global_max = -1
    t_pair = sparsedtype
    if filename.endswith('gz'):
        f=gzip.open(filename,'r')
    else:
        f=open(filename,'r')

#    t1 = time()
    try:
        for i,line in enumerate(f):
            tokens = line.split('#')[0].rstrip().split()
            label = float(tokens[0])
            labels.append(label)
            del tokens[0]
            if tokens[0].startswith("qid"):
                qids.append(int(tokens[0].split(":")[1]))
                del tokens[0]
            tokens=[(int(t[0]),float(t[1]))
                    for t in (t.split(':')
                              for t in tokens if t != '')]
            a=np.array(tokens, dtype=t_pair)
            local_max = 0.0
            if a.shape[0]>0:
                local_max = a['f0'].max()
            if local_max > global_max:
                global_max = local_max
            examples.append(a)
        return np.array(examples,dtype=np.object),np.array(labels), global_max+1
    finally:
        f.close()
        #print "data loaded in %f sec" % (time()-t1)

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
