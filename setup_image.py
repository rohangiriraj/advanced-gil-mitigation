# setup_image.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "image_cython",
        ["image_cython.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
