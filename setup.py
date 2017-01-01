import numpy as np

from setuptools import setup
from Cython.Build import cythonize

setup(
    name='netstat',
    version='0.1',
    description='Data Mining project',
    author='Mikhail Mishin, Max Reuter',
    packages=[
      'netstat',
      'netstat/graph'
    ],
    ext_modules=cythonize('netstat/graph/graph.pyx'),
    include_dirs=[np.get_include()],
    entry_points={
          'console_scripts': [
              'netstat = netstat.__main__:main',
              'clean = netstat.clean:main'
          ]}
)