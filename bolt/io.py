import sys
import numpy as np
import gzip


"""
"""
sparsedtype = np.dtype("u4,f4")
densedtype = np.float32

def dense2sparse(x):
    return np.array([(nnz, x[nnz]) for nnz in x.nonzero()[0]],dtype = sparsedtype)

class MemoryDataset:
    """A dataset implementation loads the data into memory. 
    """
    def __init__(self, dim, instances, labels):
        assert len(instances) == len(labels)
        self.dim = dim
        self.n = len(instances)
        self.instances = instances
        self.labels = labels
        self._shuffle = np.arange(self.n)
        self.classes = np.unique1d(labels)

    def __iter__(self):
        for i in self._shuffle:
            yield (self.instances[i],self.labels[i])

    def iterinstances(self):
        for i in self._shuffle:
            yield self.instances[i]

    def iterlabels(self):
        for i in self._shuffle:
            yield self.labels[i]

    def shuffle(self, seed = None):
        rs = np.random.RandomState()
        rs.seed(seed)
        rs.shuffle(self._shuffle)

    @classmethod
    def load(cls, data_file, desc = "training", verbose = 1):
        if verbose > 0:
            print "loading %s data ..." % desc,
        sys.stdout.flush()
        try:
            dim, instances, labels = loadData(data_file)
        except IOError, (errno, strerror):
            if verbose > 0:
                print(" [fail]")
            raise Exception, "cannot open '%s' - %s." % (data_file,strerror)
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
        return np.array(instances,dtype=np.object),np.array(labels), global_max+1
    finally:
        f.close()
        
def svmlToNpy():
    if sys.argv < 3 or "--help" in sys.argv:
	print """Usage: %s in-file out-file

	Converts the svm^light encoded in-file into the binary encoded out-file.
	""" % "svml2npy"
	sys.exit(-2)
    in_filename, out_filename = sys.argv[1:]
    
    instances, labels, dim = loadDat(in_filename)
    f = open(out_filename,'w+b')
    try:
        np.save(f,instances)
	np.save(f,labels)
	np.save(f,dim)
    finally:
        f.close()
