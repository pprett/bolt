from distutils.core import setup
from distutils.extension import Extension
import os.path

import numpy
numpy_path = os.path.join(numpy.__path__[0], 'core', 'include')

setup(
    name = "bolt",
    license = "MIT",
    ext_modules = [Extension("bolt/bolt", ["bolt/bolt.c"], extra_link_args=["-O3", "-ffast-math"], libraries=['profiler'], library_dirs=['/usr/local/lib'],
)],
    version = "0.1",
    description="Bolt Online Learning Toolbox",
    author='Peter Prettenhofer',
    author_email='peter.prettenhofer@gmail.com',
    packages=['bolt'],
    scripts = ["sb"],
    long_description = """Really long text here. Can be ReST""" 
)
