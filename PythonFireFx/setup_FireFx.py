# encoding: utf-8
import sys
from Cython.Build import cythonize

from distutils.core import setup
from distutils.extension import Extension

import numpy

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


extra_compile_args = []

if sys.platform == 'win32':
    extra_compile_args = ["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"]

elif sys.platform == 'darwin':
    pass

elif sys.platform == 'linux2':
    pass

elif sys.platform == 'cygwin':
    pass


# /O2 sets a combination of optimizations that optimizes code for maximum speed.
# /Ot (a default setting) tells the compiler to favor optimizations for speed over
# optimizations for size.
# /Oy suppresses the creation of frame pointers on the call stack for quicker function calls.
setup(
    name='SHADER',
    ext_modules=cythonize([
        Extension("FireFx", ["FireFx.pyx"],
                  extra_compile_args=extra_compile_args,
                  language="c"),
    ]),


    include_dirs=[numpy.get_include(), '../Include'],

)
