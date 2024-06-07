# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension("cosine_similarity_fast", ["cosine_similarity_fast.pyx"],
              include_dirs=[numpy.get_include()])  # Ensures NumPy headers are available
]

setup(
    name="Cosine Similarity Fast",
    ext_modules=cythonize(extensions)
)
