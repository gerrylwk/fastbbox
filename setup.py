# CYTHON BUILD CONFIGURATION
# This setup.py is for the stable Cython-based build of fastbbox.
# For the nanobind build, use: pip install . -C--config-file=pyproject.nanobind.toml
# See BUILD.md for more details.

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension modules
extensions = [
    Extension(
        "fastbbox.bbox",
        ["src/fastbbox/bbox.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    ),
    Extension(
        "fastbbox.obb_bbox",
        ["src/fastbbox/obb_bbox.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'nonecheck': False,
        'language_level': 3
    }),
    zip_safe=False,
)

