from distutils.core import setup
from distutils.extension import Extension
import os.path

import numpy
numpy_path = os.path.join(numpy.__path__[0], 'core', 'include')
# use additional compiler flags: "-ffast-math" "-g"

from distutils.sysconfig import get_python_inc
incdir = os.path.join(get_python_inc(plat_specific=1))

# FIXME proper win/unix lib/includes.
unix_include_path = "/usr/include"
win_include_path = "C:\Python26\include"
include_path = unix_include_path
win_lib_path = "C:\Python26\PCbuild"
unix_lib_path = "/usr/lib"
lib_path = unix_lib_path

setup(
    name = "bolt",
    license = "MIT",
    ext_modules = [
                   Extension("bolt.trainer.sgd", ["bolt/trainer/sgd.c"],
                             include_dirs=[incdir,numpy_path],
                             extra_link_args=["-O3"],
                             library_dirs=[lib_path,],
                             extra_compile_args=["-O3","-g"]
                             ),
                   Extension("bolt.trainer.avgperceptron", ["bolt/trainer/avgperceptron.c"],
                             include_dirs=[incdir,numpy_path],
                             extra_link_args=["-O3"],
                             library_dirs=[lib_path,],
                             extra_compile_args=["-O3","-g"]
                             ),
                   Extension("bolt.trainer.maxent", ["bolt/trainer/maxent.c"],
                             include_dirs=[incdir,numpy_path],
                             extra_link_args=["-O3"],
                             library_dirs=[lib_path,],
                             extra_compile_args=["-O3","-g"]
                             ),
                   ],
    version = "1.4",
    description="Bolt Online Learning Toolbox",
    author='Peter Prettenhofer',
    author_email='peter.prettenhofer@gmail.com',
    url = 'http://pprett.github.com/bolt/',
    license = 'new BSD'
    packages=['bolt','bolt.trainer', 'bolt.sandbox'],
    scripts = ["sb","sb_cv","svml2npy"],
    long_description = """Bolt features online learning algorithms
to train (generalized) linear models. Bolt is aimed at
large-scale, high-dimensional, and sparse machine learning
problems encountered in natural language processing
and information retrieval. """,
    zip_safe=False,
    classifiers = 
            ['Intended Audience :: Science/Research',
             'Intended Audience :: Developers',
             'License :: OSI Approved',
             'Programming Language :: C',
             'Programming Language :: Python',
             'Topic :: Scientific/Engineering',
             'Operating System :: Microsoft :: Windows',
             'Operating System :: POSIX',
             'Operating System :: Unix',
             'Operating System :: MacOS'
             ]
)

