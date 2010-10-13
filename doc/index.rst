..  Bolt documentation master file, created by
   sphinx-quickstart on Sat Mar  6 20:41:04 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bolt Online Learning Toolbox
============================

:Release: |version|
:Date: |today|
:Author: peter.prettenhofer@gmail.com

**Contents**:

.. toctree::
   :maxdepth: 1

   install.rst              
   overview.rst                
   using-cli.rst
   using-api.rst
   whatsnew.rst 

**Introduction**

Bolt features discriminative learning of linear predictors (e.g. `SVM <http://en.wikipedia.org/wiki/Support_vector_machine>`_ or
`Logistic Regression <http://en.wikipedia.org/wiki/Logistic_regression>`_) using fast online learning algorithms. Bolt is
aimed at large-scale, high-dimensional and sparse machine-learning problems.
In particular, problems encountered in information retrieval and
natural language processing.

Bolt considers linear models (:class:`bolt.model.LinearModel`) for binary classification, 

.. math::

   f(\mathbf{x}) = \operatorname{sign}(\mathbf{w}^T \mathbf{x} + b) , 

and generalized linear models (:class:`bolt.model.GeneralizedLinearModel`) for multi-class classification, 

.. math::

   f(\mathbf{x}) = \operatorname*{arg\,max}_y \mathbf{w}^T \Phi(\mathbf{x},y) + b_y .

Where :math:`\mathbf{w}` and :math:`b` are the model parameters that are learned from training data. 
In Bolt the model parameters are learned by minimizing the regularized training error given by, 

.. math::
    
    E(\mathbf{w},b) = \sum_{i=1}^n L(y_i,f(\mathbf{x}_i)) + \lambda R(\mathbf{w}) , 

where :math:`L` is a loss function that measures model fit and :math:`R` is a regularization term that measures model complexity. 

**Features**

Bolt supports the following trainers for binary classification: 

   * Stochastic Gradient Descent (:class:`bolt.trainer.sgd.SGD`)
        * Supports various loss functions :math:`L` : Hinge, Modified Huber, Log. 
        * Supports various regularization terms :math:`R` : L2, L1, and Elastic Net. 

   * PEGASOS (:class:`bolt.trainer.sgd.PEGASOS`)

For multi-class classification: 
   
   * One-versus-all (:class:`bolt.trainer.OVA`)
   * Averaged Perceptron (:class:`bolt.trainer.avgperceptron.AveragedPerceptron`)
   * Maximum Entropy (:class:`bolt.trainer.maxent.MaxentSGD`)
        * aka Multinomial Logistic Regression
        * Trained via SGD. 

**Benchmark**

The following RCV1-CCAT benchmark results show that Bolt is competitive to state-of-the-art linear SVM solvers such as SVM\ :sup:`Perf`, `liblinear <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_, or `sgd <http://leon.bottou.org/projects/sgd/>`_. The dataset comprises 781.264 training documents, each represented by a 47.152 dimensional feature vector. 

========================  ====================  ========
Algorithm                 Training time         Accuracy
========================  ====================  ========
SVM\ :sup:`light`         >600.00 sec          
------------------------  --------------------  --------
SVM\ :sup:`Perf` [#f1]_   11.60 sec             94.79
------------------------  --------------------  --------
liblinear [#f2]_          9.00 sec              94.77 
------------------------  --------------------  --------
bolt [#f3]_               2.33 sec              94.79
------------------------  --------------------  --------
sgd [#f4]_                1.09 sec              94.77
========================  ====================  ========

.. [#f1] Uses C=1000
.. [#f2] Uses SVM (Dual), B=1
.. [#f3] Uses E=5, r=0.00001, l=0, b
.. [#f4] Uses epochs=5, lambda=0.00001


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

