import numpy as np

from setuptools import setup
from Cython.Build import cythonize

setup(
    name='newstat',
    version='0.1',
    description='Data Mining project',
    author='Mikhail Mishin, Max Reuter',
    packages=[
      'netstat',
      'netstat/graph'
    ],
    ext_modules=cythonize('netstat/graph/graph.pyx'),
    include_dirs=[np.get_include()]
)