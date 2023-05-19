from distutils.core import setup, Extension
import numpy

setup(name='c_doc2vecc',
      version='1.0',
      ext_modules=[
          Extension
          ('c_doc2vecc',
           ['doc2vecc_pymodule.c'],
           include_dirs=[numpy.get_include()])
      ],
      )
