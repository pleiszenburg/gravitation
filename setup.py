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

import importlib
import os
import sys

from setuptools import setup

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SETUP
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class _DiscoveryError(Exception):
    pass


def _discover_ext_modules_in_kernel(kernel: str) -> list:
    "discover ext modules in kernel"

    try:
        module = importlib.import_module(f"{kernel:s}.setup")
    except ModuleNotFoundError as e:
        raise _DiscoveryError() from e

    try:
        ext_modules = module.EXTENTIONS
    except AttributeError as e:
        raise _DiscoveryError() from e

    return ext_modules


def _discover_ext_modules() -> list:
    "go through all kernels and discover ext modules"

    path = os.path.join(os.path.dirname(__file__), "src", "gravitation", "kernel")
    sys.path.insert(0, path)  # HACK

    ext_modules = []

    for kernel in os.listdir(path):
        try:
            ext_modules.extend(_discover_ext_modules_in_kernel(kernel))
        except _DiscoveryError:
            pass

    return ext_modules


setup(ext_modules=_discover_ext_modules())
