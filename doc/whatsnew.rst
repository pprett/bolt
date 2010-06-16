.. _whatsnew:

What's new
==========

Version 1.4
-----------

  * Averaged Perceptron (:class:`bolt.trainer.avgperceptron.AveragedPerceptron`)
      Multi-class classification using the averaged perceptron by Freund and Schapire. 
  * Maximum Entropy (:class:`bolt.trainer.maxent.MaxentSGD`)
      Multinomial logistic regression classifier using SGD. Supports probabilistic output. 
  * Windows compatibility
      Tested on Windows 7 using Enthought Python Distribution.

Version 1.3
-----------

  * Elastic-net penalty
      Sparse regularization of correlated predictors. 
  * PEGASOS (:class:`bolt.trainer.sgd.PEGASOS`)
      New SVM solver based on projected (sub-)gradients. 
  * Generalized Linear Models (:class:`bolt.model.GeneralizedLinearModel`)
      Generalization of linear models to multi-class classification.
  * OVA (:class:`bolt.trainer.OVA`)
      A one-versus-all approach to multi-class classification. 
  * Profiling of Cython code
      Substantial performance improvement through profiling of Cython code. 
  * Major refactoring
      changed the structure of the sub-modules (i.e., moved the SGD code to the :mod:`bolt.trainer` module) and introduced the :class:`bolt.io.Dataset` interface in :mod:`bolt.io`. 

Version 1.2
-----------

  * Cross-validation script added. 

Version 1.1
-----------

  * Bugfix release

Version 1.0
-----------

  * Initial Release