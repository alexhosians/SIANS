from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

import platform
#####
# Compile as
# python setup.py build_ext --inplace
#####
# The GSL include and libdir may have to be changed depending on where it is installed.
# Do this manually in their respective variables.
# gsl_libdir = lib directory for /usr/gsl/lib/
# gsl_incdir = include directory for /usr/gsl/include/
# libs = lib names without .lib prefix
####
# For polyhedron simulations, higher precision may be required.
# The potential may diverge if high precision variables are not used.
# This utilizes mpir to allocate more bits to variables 
# required in the polyhedron potential.
# If large polyhedra are used, it is recommended to enable high precision computation.
# To do this, compile the setup.py bile in the cymodulesmpir folder, 
# and enable mpir for the simulations

if platform.system() == 'Windows':
	# For windows users
	gsl_libdir = "C:/GSL/gsl_2_2_msvc2015_32/msvc2015_32/lib/gsl/"
	gsl_incdir = "C:/GSL/gsl_2_2_msvc2015_32/msvc2015_32/include/"
	libs = ["gsl", "cblas"]
	openmp_arg = ["/openmp"]
elif platform.system() == 'Linux':
	# For Linux users
	gsl_libdir = "/usr/lib/"
	gsl_incdir = "/usr/include/"
	libs = ["gsl", "gslcblas"]
	openmp_arg = ["-fopenmp"]
elif platform.system() == 'Darwin':
	# For Mac users
	gsl_libdir = "/usr/local/lib/"
	gsl_incdir = "/usr/local/include/"
	libs = ["gsl", "gslcblas"]
	# For homebrew users
	openmp_arg = ["-Xpreprocessor" "-fopenmp"]

ext_modules = [
	Extension("other_functions_cy",  ["other_functions_cy.pyx", "SurfaceIntegrals.c", "potentials.c", "commonCfuncs.c"], 
		libraries=libs,
		extra_compile_args=openmp_arg,
		extra_link_args=openmp_arg,
		library_dirs=[gsl_libdir],
		include_dirs=[gsl_incdir, numpy.get_include()],
		define_macros=[('CYTHON_TRACE', '1')]
		),

	Extension("ODESolvers", ["ODESolvers.pyx", "commonCfuncs.c", "diffeqsolve.c", "SurfaceIntegrals.c", "potentials.c"],
		libraries=libs,
		library_dirs=[gsl_libdir],
		include_dirs=[gsl_incdir, numpy.get_include()],
		extra_compile_args=openmp_arg,
		extra_link_args=openmp_arg,
		define_macros=[('CYTHON_TRACE', '1')]
		),

	Extension("odetableaus", ["odetableaus.pyx"],
		include_dirs=[numpy.get_include()]
		),
]

setup(
	ext_modules=cythonize(ext_modules, language_level="3")
)

