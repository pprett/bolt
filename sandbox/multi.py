"""

"""
import numpy as np
import bolt

from time import time

class BinaryDataset(bolt.io.Dataset):
    """A `Dataset` wrapper which masks the class labels. 
    """
    def __init__(self,dataset,c):
        """
        Parameters
        dataset: The `Dataset` to wrap.
        c: The positive class. 
        """
        self._dataset = dataset
        self.mask = lambda y: 1 if y == c else -1
        self.n = dataset.n
        self.dim = dataset.dim
        self.classes = np.array([1,-1],dtype=np.float32)

    def __iter__(self):
        return ((x,self.mask(y)) for x,y in self._dataset)

    def iterinstances(self):
        return self._dataset.iterinstances()

    def iterlabels(self):
        return (self.mask(y) for y in self.iterlabels)

    def shuffle(self, seed = None):
        self._dataset.shuffle(seed)

class MLinearModel(object):
    """A generalized linear model of the form z = max_y w * f(x,y) + b_y.
    """

    def __init__(self, m, k):
        """Create a generalized linear model for classification problems with `k` classes. 

        Parameters:
        m: The dimensionality of the input data (i.e., the number of features).
        k: The number of classes.
        """
        if m <= 0:
            raise ValueError, "Number of dimensions must be larger than 0."
        if k <= 2:
            raise ValueError, "Number of classes must be larger than 2 (if 2 use binary model)"
        self.m = m
        self.k = k
        self.W = np.zeros((k,m),dtype=np.float64)
        self.b = np.zeros((k,),dtype=np.float64)


    def __call__(self,x):
        """Predicts the class for the instance `x`. 
        
        Return:
        -------
        class
        """
        return self._predict(x)
            

    def predict(self,instances):
        """Predicts class of each instances in
        `instances`.

        Parameters:
        examples: a sequence of instances

        Return:
        A generator over the predictions.
        """
        for x in instances:
            yield self.__call__(x)

    def _predict(self,x):
        ps = np.array([bolt.predict(x, self.W[i], self.b[i]) for i in range(self.k)])

        #ps = np.array([sum([self.W[i,j]*v for j,v in x])+self.b[i] for i in range(self.k)])
        c = np.argmax(ps)
        return c

class OVA(object):
    """A One-versus-All trainer for multi-class models.

    It trains one binary classifier for each class `c`
    that predicts the class or all-other classes.
    """

    def __init__(self, loss = bolt.ModifiedHuber(), reg = 0.00001, biasterm = True):
        self.loss = loss
        self.reg = reg
        self.biasterm = biasterm

    def train(self, model, dataset, epochs = 5, verbose = 1, shuffle = False):
        classes = dataset.classes
        assert model.k == len(classes)
        sgd = bolt.SGD(epochs)
        t1 = time()
        for i,c in enumerate(classes):
            bmodel = bolt.LinearModel(model.m,loss = self.loss, reg = self.reg,
                                      biasterm = self.biasterm)
            dtmp = BinaryDataset(dataset,c)
            sgd.train(bmodel,dtmp, verbose = 0,
		      shuffle = shuffle)
            model.W[i] = bmodel.w.T
            model.b[i] = bmodel.bias
            if verbose > 0:
                print("Model %d trained. \nTotal training time %f seconds. " % (i,time() - t1))

def main(args):
    from nltk.corpus import news
    import nltk

    ftrain = args[0]
    ftest = args[1]

    #dim, itrain, ltrain = bolt.io.loadNpz(ftrain)
    #dim, itest, ltest = bolt.io.loadNpz(ftest)

    dtrain = bolt.io.MemoryDataset.load(ftrain, verbose = 0)
    dtest = bolt.io.MemoryDataset.load(ftest, verbose = 0)
    assert dtrain.dim == dtest.dim

    cats = news.categories()
    cats = dict(((i,c) for i,c in enumerate(news.categories())))
    k = len(cats)
    
    model = MLinearModel(dtrain.dim,k)
    ova = OVA(loss = bolt.ModifiedHuber(), reg = 0.0001)
    ova.train(model,dtrain, epochs = 20)

    ref = [cats[y] for y in dtest.iterlabels()]
    pred = [cats[z] for z in model.predict(dtest.iterinstances())]
    cm = nltk.metrics.ConfusionMatrix(ref, pred)

    print "\n"
    print "="*30
    print "Evaluation".center(30)
    print "="*30
    print "\n"
    print cm.pp()
    
    print "Accuracy: ", nltk.metrics.accuracy(ref, pred)

    print "\n"

    gold = dict([(c,[]) for c in cats.values()])
    for i,c in enumerate(ref):
        gold[c].append(i)

    
    outs = dict([(c,[]) for c in cats.values()])
    for i,c in enumerate(pred):
        outs[c].append(i)

    f1s = []
    print "%s\t%s" % ("Category".ljust(25),"F1".center(6))
    print "-"*38
    for c in cats.values():
        f1 = nltk.metrics.f_measure(set(gold[c]),set(outs[c]))
        print "%s\t%.4f" % (c.ljust(25),f1)
        f1s.append(f1)
    print "-"*38
    print "%s\t%.4f" % ("Mean".ljust(25), np.mean(f1s))

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
