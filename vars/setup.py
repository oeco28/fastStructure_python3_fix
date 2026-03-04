from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

gsl_include = os.environ.get('GSL_INCLUDE', '/usr/local/include')
gsl_lib     = os.environ.get('GSL_LIB',     '/usr/local/lib')

include_dirs = [numpy.get_include(), '.', gsl_include]
library_dirs = [gsl_lib]
libraries    = ['gsl', 'gslcblas']

# Each extension includes its companion C_*.c file so that the C symbols
# (Q_update, P_update_simple, P_update_logistic, etc.) are compiled and
# linked directly into the .so — no separate shared library needed.
ext_modules = [
    Extension("utils",
              ["utils.pyx"],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=libraries),
    Extension("admixprop",
              ["admixprop.pyx", "C_admixprop.c"],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=libraries),
    Extension("allelefreq",
              ["allelefreq.pyx", "C_allelefreq.c"],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=libraries),
    Extension("marglikehood",
              ["marglikehood.pyx", "C_marglikehood.c"],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=libraries),
]

setup(
    name='fastStructure_vars',
    ext_modules=cythonize(ext_modules,
                          include_path=['.'],
                          compiler_directives={'language_level': '3'}),
)
