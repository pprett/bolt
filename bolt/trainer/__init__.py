"""
The :mod:`trainer` package contains concrete
`Trainer` classes which are used
to train a `Model` on a `Dataset`. 

"""

__authors__ = [
      '"Peter Prettenhofer" <peter.prettenhofer@gmail.com>'
]

import sgd
import avgperceptron

from time import time

from ..model import LinearModel
from ..io import BinaryDataset

class OVA(object):
    """A One-versus-All trainer for multi-class models.

    It trains one binary classifier for each class `c`
    that predicts the class or all-other classes.
    """
    
    trainer = None
    """The concrete trainer for the binary classifiers. """

    def __init__(self, trainer):
        """
        :arg trainer: A concrete `Trainer` implementation which is used to train `k` `LinearModel` classifiers that try to predict one class versus all others. 
        """
        self.trainer = trainer
        """:member trainer: the trainer... """

    def train(self, glm, dataset, verbose = 1, shuffle = False):
        """Train the `glm` using `k` binary `LinearModel` classifiers by
        applying the One-versus-All multi-class strategy.

        :arg glm: A :class:`bolt.model.GeneralizedLinearModel`.
        :arg dataset: A `Dataset`.
        :arg verbose: Verbose output.
        :type verbose: int
        :arg shuffle: Whether to shuffle the training data; argument is passed to `OVA.trainer`.
        :type shuffle: bool
        """
        classes = dataset.classes
        assert glm.k == len(classes)
        t1 = time()
        for i,c in enumerate(classes):
            bmodel = LinearModel(glm.m, biasterm = glm.biasterm)
            dtmp = BinaryDataset(dataset,c)
            self.trainer.train(bmodel,dtmp, verbose = 0,
                               shuffle = shuffle)
            glm.W[i] = bmodel.w.T
            glm.b[i] = bmodel.bias
            if verbose > 0:
                print("Model %d trained. \nTotal training time %.2f seconds. " % (i,time() - t1))
