from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

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
    cmdclass = {'build_ext': build_ext},
    scripts = ["sb"],
    long_description = """Really long text here. Can be ReST""" 
)
