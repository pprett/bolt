"""
The :mod:`trainer` package contains concrete
`Trainer` classes which are used
to train a `Model` on a `Dataset`. 

"""

__authors__ = [
      '"Peter Prettenhofer" <peter.prettenhofer@gmail.com>'
]
import time
import sgd

from ..model import LinearModel
from ..io import BinaryDataset

class OVA(object):
    """A One-versus-All trainer for multi-class models.

    It trains one binary classifier for each class `c`
    that predicts the class or all-other classes.
    """

    def __init__(self, loss = sgd.ModifiedHuber(),
                 reg = 0.00001, biasterm = True,
                 epochs = 5):
        self.loss = loss
        self.reg = reg
        self.biasterm = biasterm
        self.epochs = epochs

    def train(self, glm, dataset, verbose = 1, shuffle = False):
        classes = dataset.classes
        assert glm.k == len(classes)
        sgd = sgd.SGD(self.loss, self.reg, self.epochs)
        t1 = time()
        for i,c in enumerate(classes):
            bmodel = LinearModel(glm.m, biasterm = self.biasterm)
            dtmp = BinaryDataset(dataset,c)
            sgd.train(bmodel,dtmp, verbose = 0,
		      shuffle = shuffle)
            glm.W[i] = bmodel.w.T
            glm.b[i] = bmodel.bias
            if verbose > 0:
                print("Model %d trained. \nTotal training time %.2f seconds. " % (i,time() - t1))
