from distutils.core import setup, Extension

setup(name='c_doc2vecc', version='1.0', ext_modules=[Extension('c_doc2vecc', ['doc2vecc_pymodule.c'])])