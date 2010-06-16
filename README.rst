Bolt Online Learning Toolbox
============================

Bolt features discriminative learning of linear predictors (e.g. `SVM <http://en.wikipedia.org/wiki/Support_vector_machine>`_ or
`Logistic Regression <http://en.wikipedia.org/wiki/Logistic_regression>`_) using fast online learning algorithms. Bolt is
aimed at large-scale, high-dimensional and sparse machine-learning problems.
In particular, problems encountered in information retrieval and
natural language processing.

Bolt features: 

   * Fast learning based on stochastic gradient descent (plain and via projected (sub-)gradients). 

   * Different loss functions for classification (hinge, log, modified huber) and regression (OLS, huber). 

   * Different penalties (L2, L1, and elastic-net). 

   * Simple, yet powerful commandline interface similar to SVM^light.

   * Python bindings, feature vectors encoded as Numpy arrays. 

Furthermore, Bolt provides preliminary support for generalized linear models for multi-class classification. Currently, it supports the following multi-class learning algorithms: 

   * One-versus-All strategy for binary classifiers.
 
   * Multinomial Logistic Regression (aka MaxEnt) via SGD.

   * Averaged Perceptron [Freund, Y. and Schapire, R. E., 1998].

The toolkit is written in Python [1], the critical sections are
C-extensions written in Cython [2]. It makes heavy use of Numpy [3], a
numeric computing library for Python. 

Requirements
------------

To install Bolt you need:

   * Python 2.5 or 2.6
   * C-compiler (tested with gcc 4.3.3)
   * Numpy (tested with 1.2.1)

If you want to modify *.pyx files you also need cython (>=0.11.2).

Installation
------------

To clone the repository run, 

   git clone git://github.com/pprett/bolt.git

To build bolt simply run,

   python setup.py build

To install bolt on your system, use

   python setup.py install

Documentation
-------------

For detailed documentation see http://pprett.github.com/bolt/.

References
----------

[1] http://www.python.org

[2] http://www.cython.org

[3] http://numpy.scipy.org

[Freund, Y. and Schapire, R. E., 1998] Large margin classification 
using the perceptron algorithm. In Machine Learning, 37, 277-296.

[Shwartz, S. S., Singer, Y., and Srebro, N., 2007] Pegasos: Primal
estimated sub-gradient solver for svm. In Proceedings of ICML '07.

[Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009] Stochastic gradient
descent training for l1-regularized log-linear models with cumulative
penalty. In Proceedings of the AFNLP/ACL '09.

[Zhang, T., 2004] Solving large scale linear prediction problems using
stochastic gradient descent algorithms. In Proceedings of ICML '04.

[Zou, H., and Hastie, T., 2005] Regularization and variable selection via 
the elastic net. Journal of the Royal Statistical Society Series B, 
67 (2), 301-320.

