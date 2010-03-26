.. _using:

Using Bolt
==========

:Release: |version|
:Date: |today|


Binary classification
---------------------

To use Bolt for binary classification we first need some data. Bolt is aimed primarily at high-dimensional and sparse machine learning problems. Thus, instances are represented as sparse vectors. Bolt uses numpy record arrays to represent sparse vectors. The following example shows an instance `x` with features `0` and `4` set to `1` and `0.2`, respectively. All other features are assumed to be zero. ::

  x = np.array([(0,1),(4,0.2)], dtype = bolt.sparsedtype)

For binary classification, Bolt assumes that the class labels, either positive or negative, are coded as `y = 1` or `y = -1`. 

Lets create some synthetic dataset to show how to use Bolt for binary classification...

For convenience, Bolt offers routines to load datasets in binary or svm^light format via the :mod:io module. ::

  dataset = bolt.io.MemoryDataset.load("train.dat.gz")




Multi-class classification
--------------------------

Bolt supports multi-class classification via **generalized linear models** (GLM). Currently, the only multi-class trainer is :class:`bolt.trainer.OVA` which trains `k` binary classifiers, where `k` is the number of different classes. At test time it predicts the class with the highest confidence. 

First, lets get some data (e.g., the 20-newsgroups dataset): ::

  import bolt
  dtrain = bolt.io.MemoryDataset.load(ftrain, verbose = 0)
  dtest = bolt.io.MemoryDataset.load(ftest, verbose = 0)

Next, we create a :class:`bolt.model.GeneralizedLinearModel` ::

  glm = bolt.GeneralizedLinearModel(dtrain.dim,k, biasterm = False)

The GLM receives two parameters: the dimensionality of the input data and the number of classes `k`. The third parameters indicates that a class specific bias term is used. 

In order to train the `glm` with the :class:`bolt.trainer.OVA` trainer we need instantiate the base trainer which is used by :class:`bolt.trainer.OVA` to train the binary classifiers. In the following example, we will use a :class:`bolt.trainer.sgd.SGD` trainer. ::

    sgd = bolt.SGD(bolt.ModifiedHuber(), reg = 0.0001, epochs = 20)

Now, we can create a :class:`bolt.trainer.OVA` trainer and train the `glm` on the training data: ::

    ova = bolt.OVA(sgd)
    ova.train(glm,dtrain)

To get the predictions on the test data use :func:`bolt.model.GeneralizedLinearModel.predict` which gives you the index of the predicted class in `dtrain.classes`: ::

    pred = [drain.classes[z] for z in model.predict(dtest.iterinstances())]