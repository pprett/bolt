.. _install:

=======
Install
=======

Describes how to install Bolt on your machine. 

.. contents::

Requirements
------------

Bolt depends on the following python packages: 

  * numpy (tested with version 1.1 - 1.3)

To check your installed version use::

  python -c "import numpy; print numpy.__version__"

To build Bolt you need a c compiler (e.g., gcc) and the header files of `numpy`. 

On most Ubuntu systems the following should suffice::

  sudo apt-get install python-numpy

If you want to modify the Cython code in the extension module `bolt.bolt` make sure that the newest release of Cython (>= 0.7) is installed::

  sudo easy_install cython
  
Stable release
--------------

Download the latest release from github::

  wget http://github.com/pprett/bolt/tarball/v1.2 -O bolt_v1.2.tar.gz

Untar the archive::

  tar xzvf bolt_v1.2.tar.gz
  mv pprett* bolt

Build bolt::

  cd bolt
  python setup.py build

Install bolt sytem wide::

  sudo python setup.py install

If the build process fails due to undefined references make sure that the include paths in `setupy.py` are properly defined. 

If you need assistance feel free to contact the author at peter.prettenhofer@gmail.com. 

Latest development version
--------------------------

To fetch the latest development version from the repository type::

  git checkout git://github.com/pprett/bolt.git

To build the project use::

  python setup.py build

To install bolt system wide use::

  sudo python setup.py install