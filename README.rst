Bolt Online Learning Toolbox
============================

Bolt features discriminative learning of linear predictors (e.g. SVM or
logistic regression) using stochastic gradient descent. Bolt is aimed at large-scale, high-dimensional and sparse machine-learning problems. In particular, problems encountered in information retrieval and natural language processing. 

Bolt considers linear prediction problems where one wants to learn a
linear predictor f(x) which minimizes a given error function E(w,b),  

   f(x) = w*x + b

Where, x is the example to be predicted, w is a vector of parameters
(aka the weight vector) which specifiy the predictor, b is the bias
term and w*x represents the inner (dot) product of w and x. The error
function takes the following form, 

   Error = loss + penalty
   .. role:: raw-math(raw)
       :format: latex html
       E(\mathbf{w},b) = \sum_{i=1}^n L(y_i,f(x_i)) + \lambda R(\mathbf{w})

Where, 'penalty' is a term which penalizes model complexity (usually
the l2-norm of the weight vector w) and 'loss' is a
convex loss function such as hinge loss or squared error. Popular
linear models such as ridge regression, logistic regression or
(linear) support vector machines can be expressed in the above
framework.

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

If you want to modify bolt.pyx you also need cython (>=0.11.2).

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

[3] http://numpy.scipy.org/

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

