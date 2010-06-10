.. _using-cli:

Using Bolt (CLI)
================

:Release: |version|
:Date: |today|

This section shows how to use Bolt via its command line interface. First, we will introduce the CLI. Then, we will discuss the how to use Bolt via its CLI for a) binary and for b) multi-class classification. 

Command Line Interface
----------------------

Bolt provides a number of scripts which allow trainig, testing, and model selection from the command line. 

Currently, the follwoing scripts are shipped with bolt: 

  *sb*
      The main script of Bolt for training and testing. 

  *sb_cv*
         A script for model selection using cross-validation. 

  *svml2npy*
            A utility script to convert data files from textual (svm^light) to binary format.

For detailed information on the scripts see below or execute the script with the flag `--help`. 

sb
^^

::

    Usage: sb [options] example_file

    Bolt Online Learning Toolbox V1.4: Discriminative learning of linear models
    using stochastic gradient descent.

    Copyright: Peter Prettenhofer <peter.prettenhofer@gmail.com>

    This software is available for non-commercial use only. It must not
    be modified and distributed without prior permission of the author.
    The author is not responsible for implications from the use of this
    software.

    http://github.com/pprett/bolt

    Options:
      --version             show program's version number and exit
      -h, --help            show this help message and exit
      -v [0,1,2], --verbose=[0,1,2]    verbose output
      -c CLSTYPE, --clstype=CLSTYPE  Classifier type.
			    sgd: Stochastic Gradient Descent [default].
			    pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
			    ova: One-vs-All strategy for SGD classifiers.
			    maxent: Maximum Entropy (via SGD).
			    avgperc: Averaged Perceptron.
      -l [0..], --loss=[0..]
			    Loss function to use.
			    0: Hinge loss.
			    1: Modified huber loss [default].
			    2: Log loss.
			    5: Squared loss.
			    6: Huber loss.
      -r float, --reg=float
			    Regularization term lambda [default 0.0001].
      -e float, --epsilon=float
			    Size of the regression tube.
      -n [1,2,3], --norm=[1,2,3]
			    Penalty to use.
			    1: L1.
			    2: L2 [default].
			    3: Elastic Net: (1-a)L1 + aL2.
      -a float, --alpha=float
			    Elastic Net parameter alpha [requires -n 3; default
			    0.85].
      -E int, --epochs=int  Number of epochs to perform [default 5].
      --shuffle             Shuffle the training data after each epoche.
      -b, --bias            Use a biased hyperplane (w^t x + b) [default False].
      -p FILE, --predictions=FILE
			    Write predicitons to FILE. If FILE is '-' write to
			    stdout [either -t or --test-only are required].
      -t FILE               Evaluate the model on a separate test file.
      -m FILE, --model=FILE
			    If --test-only: Apply seralized model in FILE to
			    example_file.
			    else: store trained model in FILE.
      --test-only           Apply serialized model in option -m to example_file
			    [requires -m].
      --train-error         Compute training error [False].

    More details in:

    [Zhang, T., 2004] Solving large scale linear prediction problems using
    stochastic gradient descent algorithms. In ICML '04. 

    [Shwartz, S. S., Singer, Y., and Srebro, N., 2007] Pegasos: Primal
    estimated sub-gradient solver for svm. In ICML '07. 

    [Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009] Stochastic gradient
    descent training for l1-regularized log-linear models with cumulative
    penalty. In ACL '09.  

sb_cv
^^^^^

::

    Usage: sb_cv [options] example_file

    Bolt Online Learning Toolbox V1.4: Discriminative learning of linear models
    using stochastic gradient descent.

    Copyright: Peter Prettenhofer <peter.prettenhofer@gmail.com>

    This software is available for non-commercial use only. It must not
    be modified and distributed without prior permission of the author.
    The author is not responsible for implications from the use of this
    software.

    http://github.com/pprett/bolt

    Options:
      --version             show program's version number and exit
      -h, --help            show this help message and exit
      -v [0,1,2], --verbose=[0,1,2]
			    verbose output
      -c CLSTYPE, --clstype=CLSTYPE
			    Classifier type.
			    sgd: Stochastic Gradient Descent [default].
			    pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
			    ova: One-vs-All strategy for SGD classifiers.
			    maxent: Maximum Entropy (via SGD).
			    avgperc: Averaged Perceptron.
      -l [0..], --loss=[0..]
			    Loss function to use.
			    0: Hinge loss.
			    1: Modified huber loss [default].
			    2: Log loss.
			    5: Squared loss.
			    6: Huber loss.
      -r float, --reg=float
			    Regularization term lambda [default 0.0001].
      -e float, --epsilon=float
			    Size of the regression tube.
      -n [1,2,3], --norm=[1,2,3]
			    Penalty to use.
			    1: L1.
			    2: L2 [default].
			    3: Elastic Net: (1-a)L1 + aL2.
      -a float, --alpha=float
			    Elastic Net parameter alpha [requires -n 3; default
			    0.85].
      -E int, --epochs=int  Number of epochs to perform [default 5].
      --shuffle             Shuffle the training data after each epoche.
      -b, --bias            Use a biased hyperplane (w^t x + b) [default False].
      -f int, --folds=int   number of folds [default 10].
      -s int, --seed=int    seed for CV shuffle [default none].

    More details in:

    [Zhang, T., 2004] Solving large scale linear prediction problems using
    stochastic gradient descent algorithms. In ICML '04. 

    [Shwartz, S. S., Singer, Y., and Srebro, N., 2007] Pegasos: Primal
    estimated sub-gradient solver for svm. In ICML '07. 

    [Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009] Stochastic gradient
    descent training for l1-regularized log-linear models with cumulative
    penalty. In ACL '09.  


svml2npy
^^^^^^^^
::

   Usage: svml2npy in-file out-file

        Converts the svm^light encoded in-file into the binary encoded out-file.



Binary classification
---------------------




Multi-class classification
--------------------------

