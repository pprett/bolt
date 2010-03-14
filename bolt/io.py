"""
Input/Output module
===================

An instance in Bolt is represented as a sparse vector via a numpy record array.
The data type of the record array is `sparsedtype` which is a tuple (uint32,float64).

Classes
-------

-`MemoryDataset`, an in-memory dataset. 

"""

import sys
import numpy as np
import gzip

from itertools import izip

"""
"""
sparsedtype = np.dtype("u4,f4")
densedtype = np.float32

def dense2sparse(x):
    return np.array([(nnz, x[nnz]) for nnz in x.nonzero()[0]],dtype = sparsedtype)

class Dataset(object):
    """Dataset interface.
    """

    def __iter__(self):
        pass

    def iterinstances(self):
        pass

    def iterlabels(self):
        pass

    def shuffle(self, seed = None):
        pass

    def split(self,folds):
	pass

class MemoryDataset(Dataset):
    """An in-memory dataset.
    The instances and labels are stored as two parallel arrays.
    Access to the parallel arrays is via an indexing array which
    allows convenient shuffeling.

    TODO: implement in Cython if CEP 307 is done.
    """
    def __init__(self, dim, instances, labels):
        assert len(instances) == len(labels)
        self.dim = dim
        self.n = len(instances)
        self.instances = instances
        self.labels = labels
        self._idx = np.arange(self.n)
        self.classes = np.unique1d(labels)
        
    def __iter__(self):
        return izip(self.instances[self._idx],self.labels[self._idx])

    def iterinstances(self):
        for i in self._idx:
            yield self.instances[i]

    def iterlabels(self):
        for i in self._idx:
            yield self.labels[i]

    def shuffle(self, seed = None):
	"""Shuffles the index array using `numpy.random.shuffle`.
	A `seed` for the pseudo random number generator can be provided.
	"""
        rs = np.random.RandomState()
        rs.seed(seed)
        rs.shuffle(self._idx)

    def split(self, nfolds):
	"""Split the `Dataset` into `nfolds` new `Dataset` objects.
	The split is done according to the index array.
	"""
	r = self.n % nfolds
	num = self.n-r
	folds = np.split(self._idx[:num],nfolds)
	dsets = []
	for fold in folds:
	    dsets.append(MemoryDataset(self.dim, self.instances[fold],
				       self.labels[fold]))
	return np.array(dsets,dtype=np.object)

    @classmethod
    def merge(cls, dsets):
	"""Merge a sequence of `Dataset` objects. 
	"""
	assert len(dsets) > 1
	instances = np.concatenate([ds.instances[ds._idx] for ds in dsets])
	labels = np.concatenate([ds.labels[ds._idx] for ds in dsets])
	return MemoryDataset(dsets[0].dim, instances, labels)        

    @classmethod
    def load(cls, fname, verbose = 1):
        """Factory method to deserialize a `Dataset`. 
        """
        if verbose > 0:
            print "loading data ...",
        sys.stdout.flush()
        try:
            dim, instances, labels = loadData(fname)
        except IOError, (errno, strerror):
            if verbose > 0:
                print(" [fail]")
            raise Exception, "cannot open '%s' - %s." % (fname,strerror)
        except Exception, exc:
            if verbose > 0:
                print(" [fail]")
            raise Exception, exc
        else:
            if verbose > 0:
                print(" [done]")

        if verbose > 1:
            print("%d (%d+%d) examples loaded. " % (len(instances),
                                                    labels[labels==1.0].shape[0],
                                                    labels[labels==-1.0].shape[0]))
        return MemoryDataset(dim, instances, labels)


    def store(self,fname):
	"""Store `Dataset` in binary form.
	Uses `numpy.save` for serialization.
	"""
	f = open(fname,'w+b')
	try:
	    np.save(f,self.instances)
	    np.save(f,self.labels)
	    np.save(f,self.dim)
	finally:
	    f.close()

def loadData(filename):
    """Load a dataset from the given filename.

    If the filename ends with '.npy' it is assumed
    to be stored in binary format as a numpy archive
    (first example array, second labels;
    finally number of dimensions).

    Else it is assumed to be stored in svm^light
    format (textual). Number of dimensions are computed
    on the fly. 
    """
    loader = None
    if filename.endswith('.npy'):
        loader = loadNpz
    else:
        loader = loadDat
    dim, instances,labels = loader(filename)
    return dim, instances, labels

def loadNpz(filename):
    f = open(filename,'r')
    try:
        instances = np.load(f)
        labels = np.load(f)
        dim = np.load(f)
        return dim, instances, labels
    finally:
        f.close()

def loadDat(filename):
    labels = []
    instances = []
    qids = []
    global_max = -1
    t_pair = sparsedtype
    if filename.endswith('gz'):
        f=gzip.open(filename,'r')
    else:
        f=open(filename,'r')

    try:
        for i,line in enumerate(f):
            tokens = line.split('#')[0].rstrip().split()
            label = float(tokens[0])
            labels.append(label)
            del tokens[0]
            if len(tokens) > 0 and tokens[0].startswith("qid"):
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
            instances.append(a)
        return global_max+1, np.array(instances,dtype=np.object),np.array(labels)
    finally:
        f.close()
        
def svmlToNpy():
    if sys.argv < 3 or "--help" in sys.argv:
	print """Usage: %s in-file out-file

	Converts the svm^light encoded in-file into the binary encoded out-file.
	""" % "svml2npy"
	sys.exit(-2)
    in_filename, out_filename = sys.argv[1:]
    
    ds = Dataset.load(in_filename)
    ds.store(out_filename)
    
