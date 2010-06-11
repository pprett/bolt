.. _using-api:

Using Bolt (API)
================

:Release: |version|
:Date: |today|

This section shows how to use Bolt via its Python API. We will discuss the use of Bolt for a) binary and for b) multi-class classification. 

Binary classification
^^^^^^^^^^^^^^^^^^^^^

To use Bolt for binary classification we first need some data. Bolt is aimed primarily at high-dimensional and sparse machine learning problems. Thus, instances are represented as sparse vectors. Bolt uses numpy record arrays to represent sparse vectors. The following example shows an instance `x` with features `0` and `4` set to `1` and `0.2`, respectively. All other features are assumed to be zero. ::

  x = np.array([(0,1),(4,0.2)], dtype = bolt.sparsedtype)

For binary classification, Bolt assumes that the class labels, either positive or negative, are coded as `y = 1` or `y = -1`. 

Lets create some synthetic dataset to show how to use Bolt for binary classification...

For convenience, Bolt offers routines to load datasets in binary or svm^light format via the :mod:`bolt.io` module. ::

  dtrain = bolt.io.MemoryDataset.load("train.dat.gz")

Now we've to crate a linear model ::

  lm = bolt.LinearModel(dtrain.dim, biasterm = False)

The first parameter is mandatory and indicates the number of features in the training data. The second parameter specifies whether or not a biasterm should be included. That is, if `biasterm = True` the model computes `y = w*x + b` else `y = w*x`. Thus, `biasterm = False` forces the hyperplane to go through the origin. In general, this limitation does not harm the classificaton performance but makes the numerical operations more stable. 

To train the model we first have to instantiate a model trainer - in this example we will use :class:`bolt.trainer.sgd.SGD` ::

  sgd = bolt.SGD(bolt.ModifiedHuber(), reg = 0.0001, epochs = 20)

The trainer receives a number of parameters, see :class:`bolt.trainer.sgd.SGD` for more information on the parameterization of stochastic gradient descent. Currently, only two trainers for binary classification are provided, :class:`bolt.trainer.sgd.SGD` and :class:`bolt.trainer.sgd.PEGASOS`. 

To train the model on `dtrain` you can simply use the :func:`bolt.trainer.sgd.SGD.train` method::

  sgd.train(lm, dtrain)

Now the model `lm` is trained; You can evaluate the model on some test data via the :mod:`bolt.eval` module. ::

  dtest = bolt.io.MemoryDataset.load("test.dat.gz")
  bolt.eval.errorrate(lm,dtest)

To inspect the model parameters, simply access the models attributes ::

  print lm.w # gives the weight vector
  print lm.b # gives the bias term


Multi-class classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Bolt supports multi-class classification via **generalized linear models** (GLM). Currently, there exists the following multi-class trainers: 

  *  :class:`bolt.trainer.OVA`
  *  :class:`bolt.trainer.avgperceptron.AveragedPerceptron`
  *  :class:`bolt.trainer.maxent.MaxentSGD`

In the following example we will use the :class:`bolt.trainer.OVA` trainer which trains `k` binary classifiers, where `k` is the number of different classes. At test time it predicts the class with the highest confidence. 

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