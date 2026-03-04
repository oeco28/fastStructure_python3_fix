from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

gsl_include = os.environ.get('GSL_INCLUDE', '/usr/local/include')
gsl_lib     = os.environ.get('GSL_LIB',     '/usr/local/lib')

include_dirs = [numpy.get_include(), '.', 'vars/', gsl_include]
library_dirs = [gsl_lib]
libraries    = ['gsl', 'gslcblas']

setup(
    name='fastStructure',
    ext_modules=cythonize(
        [
            Extension("parse_bed",
                      ["parse_bed.pyx"],
                      include_dirs=include_dirs,
                      library_dirs=library_dirs,
                      libraries=libraries),
            Extension("parse_str",
                      ["parse_str.pyx"],
                      include_dirs=include_dirs,
                      library_dirs=library_dirs,
                      libraries=libraries),
            Extension("fastStructure",
                      ["fastStructure.pyx"],
                      include_dirs=include_dirs,
                      library_dirs=library_dirs,
                      libraries=libraries),
        ],
        include_path=['.', 'vars/'],
        compiler_directives={'language_level': '3'},
    ),
)
