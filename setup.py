# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    setup.py: Used for package distribution

    Copyright (C) 2019-2023 Sebastian M. Ernst <ernst@pleiszenburg.de>

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

import os

from setuptools import (
    Extension,
    find_packages,
    setup,
)
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import sysconfig

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SETUP
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Define source directory (path)
SRC_DIR = "src"

# Prepare list of extension modules (C ...)
ext_modules = cythonize(
    [
        # Extension(
        #     "gravitation.kernel.cy1.core",
        #     [os.path.join(SRC_DIR, "gravitation", "kernel", "cy1", "core.pyx")],
        # ),
        # Extension(
        #     "gravitation.kernel.cy2.core",
        #     [os.path.join(SRC_DIR, "gravitation", "kernel", "cy2", "core.pyx")],
        # ),
        # Extension(
        #     "gravitation.kernel.cy4.core",
        #     [os.path.join(SRC_DIR, "gravitation", "kernel", "cy4", "core.pyx")],
        #     extra_compile_args=["-fopenmp"],
        #     extra_link_args=["-fopenmp"],
        # ),
    ],
    annotate=True,
) + [
    Extension(
        "gravitation.kernel._lib1_.lib",
        [os.path.join(SRC_DIR, "gravitation", "kernel", "_lib1_", "lib.c")],
        extra_compile_args=[
            "-std=gnu11",
            "-fPIC",
            "-O3",
            "-ffast-math",
            "-march=native",
            "-mtune=native",
            "-mfpmath=sse",
            "-Wall",
            "-Wdouble-promotion",
            "-Winline",
            "-Werror",
        ],
        extra_link_args=["-lm"],
    ),
    # Extension(
    #     "gravitation.kernel._lib4_.lib",
    #     [os.path.join(SRC_DIR, "gravitation", "kernel", "_lib4_", "lib.c")],
    #     extra_compile_args=[
    #         "-std=gnu11",
    #         "-fPIC",
    #         "-O3",
    #         "-ffast-math",
    #         "-march=native",
    #         "-mtune=native",
    #         "-mfpmath=sse",
    #         "-fopenmp",
    #         "-Wall",
    #         "-Wdouble-promotion",
    #         "-Winline",
    #         "-Wno-maybe-uninitialized",
    #         "-Werror",
    #     ],
    #     extra_link_args=["-lm", "-fopenmp"],
    # ),
]


# HACK https://github.com/cython/cython/issues/1740#issuecomment-317556084
def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if ext_suffix == ext:
        return filename
    ext_suffix = ext_suffix.replace(ext, "")
    idx = name.find(ext_suffix)
    if idx == -1:
        return filename
    else:
        return name[:idx] + ext


class build_ext_custom(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        if filename.startswith("lib"):
            return get_ext_filename_without_platform_suffix(filename)
        else:
            return filename


# Install package
setup(
    packages=find_packages(SRC_DIR),
    package_dir={"": SRC_DIR},
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": build_ext_custom,
    },
)
