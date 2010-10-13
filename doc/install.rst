.. _install:

=======
Install
=======

Describes how to install Bolt on your machine. 

.. contents::

Requirements
------------

Bolt depends on the following python packages: 

  * numpy (tested with version 1.1 - 1.5)

To check your installed version use::

  python -c "import numpy; print numpy.__version__"

To build Bolt you need a c compiler (e.g., gcc) and the header files of `numpy`. 

On Ubuntu systems the following should suffice::

  sudo apt-get install python-numpy

The installation has been tested on Windows 7 using the `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_.
If you get undefined symbol errors, make sure that the lib_dir path in `setup.py` is properly set (for EPD 6.2 use `C:\\Python26\\PCbuild`).
 
If you want to modify the Cython code in the extension modules in `bolt.trainer` make sure that the newest release of `Cython <http://www.cython.org/>`_ (>= 0.7) is installed. 

To generate the documentation you need `Sphinx <http://sphinx.pocoo.org/>`_ and the following packages

  * `dvipng <http://savannah.nongnu.org/projects/dvipng/>`_ .
  * `sphinx-to-github <http://github.com/michaeljones/sphinx-to-github/>`_ .

On Ubuntu systems the following should suffice::

  sudo easy_install sphinx
  sudo apt-get install dvipng
  pip install -e git+git://github.com/michaeljones/sphinx-to-github.git#egg=sphinx-to-github

If you don't have `pip` install it via::

  sudo easy_install pip
  
Stable release
--------------

Download the latest release from github::

  wget http://github.com/pprett/bolt/tarball/v1.4 -O bolt_v1.4.tar.gz

Untar the archive::

  tar xzvf bolt_v1.4.tar.gz
  mv pprett* bolt

Build bolt::

  cd bolt
  python setup.py build

Install bolt sytem wide::

  sudo python setup.py install

If the build process fails due to undefined references make sure that the include paths in `setupy.py` are properly defined. 

If you need assistance please contact the author at peter.prettenhofer@gmail.com. 

Latest development version
--------------------------

To fetch the latest development version from the repository type::

  git checkout git://github.com/pprett/bolt.git

To build the project use::

  python setup.py build

To install bolt system wide use::

  sudo python setup.py install
