# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	setup.py: Used for package distribution

	Copyright (C) 2019 Sebastian M. Ernst <ernst@pleiszenburg.de>

<LICENSE_BLOCK>
The contents of this file are subject to the GNU General Public License
Version 2 ("GPL" or "License"). You may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt
https://github.com/pleiszenburg/gravitation/blob/master/LICENSE

Software distributed under the License is distributed on an "AS IS" basis,
WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License for the
specific language governing rights and limitations under the License.
</LICENSE_BLOCK>

"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from setuptools import (
	Extension,
	find_packages,
	setup,
	)
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os
import sys
import sysconfig

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SETUP
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Package version
__version__ = '0.0.2'

# List all versions of Python which are supported
confirmed_python_versions = [
	('Programming Language :: Python :: %s' % x)
	for x in '3.6 3.7'.split(' ')
	]

# Fetch readme file
with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
	long_description = f.read()

# Define source directory (path)
SRC_DIR = 'src'

# Prepare list of extension modules (C ...)
ext_modules = cythonize(
	[
		Extension(
			'gravitation.kernel.cy1.core',
			[os.path.join(SRC_DIR, 'gravitation', 'kernel', 'cy1', 'core.pyx')],
			),
		Extension(
			'gravitation.kernel.cy2.core',
			[os.path.join(SRC_DIR, 'gravitation', 'kernel', 'cy2', 'core.pyx')],
			),
		Extension(
			'gravitation.kernel.cy4.core',
			[os.path.join(SRC_DIR, 'gravitation', 'kernel', 'cy4', 'core.pyx')],
			extra_compile_args = ['-fopenmp'],
			extra_link_args = ['-fopenmp'],
			),
		],
	annotate = True,
	) + [
		Extension(
			'gravitation.kernel._lib1_.lib',
			[os.path.join(SRC_DIR, 'gravitation', 'kernel', '_lib1_', 'lib.c')],
			extra_compile_args = [
				'-std=gnu11',
				'-fPIC',
				'-O3',
				'-ffast-math',
				'-march=native',
				'-mtune=native',
				'-mfpmath=sse',
				'-Wall',
				'-Wdouble-promotion',
				'-Winline',
				'-Werror',
				],
			extra_link_args = ['-lm'],
			),
		Extension(
			'gravitation.kernel._lib4_.lib',
			[os.path.join(SRC_DIR, 'gravitation', 'kernel', '_lib4_', 'lib.c')],
			extra_compile_args = [
				'-std=gnu11',
				'-fPIC',
				'-O3',
				'-ffast-math',
				'-march=native',
				'-mtune=native',
				'-mfpmath=sse',
				'-fopenmp',
				'-Wall',
				'-Wdouble-promotion',
				'-Winline',
				'-Wno-maybe-uninitialized',
				'-Werror',
				],
			extra_link_args = ['-lm', '-fopenmp'],
			),
	]

# HACK https://github.com/cython/cython/issues/1740#issuecomment-317556084
def get_ext_filename_without_platform_suffix(filename):
	name, ext = os.path.splitext(filename)
	ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
	if ext_suffix == ext:
		return filename
	ext_suffix = ext_suffix.replace(ext, '')
	idx = name.find(ext_suffix)
	if idx == -1:
		return filename
	else:
		return name[:idx] + ext
class build_ext_custom(build_ext):
	def get_ext_filename(self, ext_name):
		filename = super().get_ext_filename(ext_name)
		if filename.startswith('lib'):
			return get_ext_filename_without_platform_suffix(filename)
		else:
			return filename

# Install package
setup(
	name = 'gravitation',
	packages = find_packages(SRC_DIR),
	package_dir = {'': SRC_DIR},
	version = __version__,
	description = 'n-body-simulation performance test suite',
	long_description = long_description,
	author = 'Sebastian M. Ernst',
	author_email = 'ernst@pleiszenburg.de',
	url = 'https://github.com/pleiszenburg/gravitation',
	download_url = 'https://github.com/pleiszenburg/gravitation/archive/v%s.tar.gz' % __version__,
	license = 'GPLv2',
	keywords = [
		'benchmark',
		'test suite',
		'n-body',
		'numerical computing',
		'high-performance computing',
		'parallelization',
		'cuda',
		'gpgpu',
		'simd',
		'openmp',
		],
	scripts = [],
	include_package_data = True,
	ext_modules = ext_modules,
	cmdclass = {
		'build_ext': build_ext_custom,
		},
	setup_requires = [
		'Cython',
		],
	install_requires = [
		'asciiplotlib',
		'click',
		# 'cupy',
		'Cython',
		'gputil',
		'h5py',
		'joblib',
		'numba',
		'numpy',
		'numexpr',
		'oct2py',
		'plotly',
		'psutil',
		# 'pycuda',
		'pygame',
		'py-cpuinfo',
		'py_mini_racer',
		# 'torch',
		],
	extras_require = {'dev': [
        'black',
		# 'pytest',
		'python-language-server',
		'setuptools',
		# 'Sphinx',
		# 'sphinx_rtd_theme',
		'twine',
		'wheel',
		]},
	zip_safe = False,
	entry_points = {
		'console_scripts': [
			'gravitation = gravitation.cli:cli',
			],
		},
	classifiers = [
		'Development Status :: 3 - Alpha',
		'Environment :: Console',
		'Intended Audience :: Developers',
		'Intended Audience :: Education',
		'Intended Audience :: Financial and Insurance Industry',
		'Intended Audience :: Healthcare Industry',
		'Intended Audience :: Information Technology',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
		'Operating System :: MacOS',
		'Operating System :: POSIX :: BSD',
		'Operating System :: POSIX :: Linux',
		'Programming Language :: Python :: 3'
		] + confirmed_python_versions + [
		'Programming Language :: Python :: 3 :: Only',
		'Programming Language :: Python :: Implementation :: CPython',
		'Programming Language :: Python :: Implementation :: PyPy',
		'Programming Language :: C',
		'Programming Language :: Cython',
		'Programming Language :: JavaScript',
		'Programming Language :: Other Scripting Engines',
		'Topic :: Education',
		'Topic :: Scientific/Engineering',
		'Topic :: Scientific/Engineering :: Astronomy',
		'Topic :: Scientific/Engineering :: Physics',
		'Topic :: Scientific/Engineering :: Visualization',
		'Topic :: Software Development',
		'Topic :: System :: Benchmark',
		'Topic :: System :: Distributed Computing',
		]
	)
