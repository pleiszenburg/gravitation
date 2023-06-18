# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/cc1/kernel.py: Kernel

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

import ctypes
import os
import sysconfig

from gravitation import BaseUniverse, typechecked

from .meta import DESCRIPTION

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Universe(BaseUniverse):
    __doc__ = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._iterate_stage1_c = None
        self._univ_free_c = None
        self._univ_c = None
        self._masses_c = None

    def start_kernel(self):
        cdtype = getattr(
            ctypes,
            f"c_{dict(float32 = 'float', float64 = 'double')[self._variation.getvalue('dtype')]:s}",
        )
        fields = ["rx", "ry", "rz", "ax", "ay", "az", "m"]

        class Mass(ctypes.Structure):
            _fields_ = [
                (field, cdtype) for field in fields
            ]

        class Univ(ctypes.Structure):
            _fields_ = [
                ('masses', ctypes.POINTER(Mass)),
                ('n', ctypes.c_size_t),
                ('g', cdtype),
            ]

        lib = ctypes.cdll.LoadLibrary(
            os.path.join(os.path.dirname(__file__), f"lib.{sysconfig.get_config_var('SOABI')}.so")
        )
        suffix = dict(float32 = 'f4', float64 = 'f8')[self._variation.getvalue('dtype')]

        univ_alloc_c = getattr(lib, f'univ_alloc_{suffix:s}')
        univ_alloc_c.argtypes = (ctypes.POINTER(Univ),)

        self._univ_free_c = getattr(lib, f'univ_free_{suffix:s}')
        self._univ_free_c.argtypes = (ctypes.POINTER(Univ),)

        self._iterate_stage1_c = getattr(lib, f'univ_iterate_stage1_{suffix:s}')
        self._iterate_stage1_c.argtypes = (ctypes.POINTER(Univ),)

        self._univ_c = Univ()
        self._univ_c.n = len(self)
        self._univ_c.g = self._G
        univ_alloc_c(ctypes.pointer(self._univ_c))

        self._masses_c = ctypes.cast(self._univ_c.masses, ctypes.POINTER(Mass * len(self))).contents
        for idx, pm in enumerate(self._masses):
            self._masses_c[idx].m = pm.m

    def stop_kernel(self):
        self._univ_free_c(ctypes.pointer(self._univ_c))

    def push_stage1(self):
        for idx, pm in enumerate(self._masses):
            (
                self._masses_c[idx].rx,
                self._masses_c[idx].ry,
                self._masses_c[idx].rz,
            ) = pm.r

    def iterate_stage1(self):
        self._iterate_stage1_c(ctypes.pointer(self._univ_c))

    def pull_stage1(self):
        for idx, pm in enumerate(self._masses):
            pm.a[:] = [
                self._masses_c[idx].ax,
                self._masses_c[idx].ay,
                self._masses_c[idx].az,
            ]
