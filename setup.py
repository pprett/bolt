from distutils.core import setup
from distutils.extension import Extension
import os.path

import numpy
numpy_path = os.path.join(numpy.__path__[0], 'core', 'include')
# use additional compiler flags: "-ffast-math" "-g"
setup(
    name = "bolt",
    license = "MIT",
    ext_modules = [Extension("bolt/bolt", ["bolt/bolt.c"],
                             include_dirs=['/usr/include/',numpy_path],
                             extra_link_args=["-O3"],
                             library_dirs=['/usr/lib','/usr/local/lib',],
                             libraries=["cblas"],
                             extra_compile_args=["-O3","-g"],)],
    version = "1.2",
    description="Bolt Online Learning Toolbox",
    author='Peter Prettenhofer',
    author_email='peter.prettenhofer@gmail.com',
    packages=['bolt','bolt.sandbox'],
    scripts = ["sb","sb_cv","svml2npy"],
    long_description = """Really long text here. Can be ReST""" 
)
