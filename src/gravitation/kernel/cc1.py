# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/cc1.py: Kernel

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
# KERNEL META
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__longname__ = "c-backend 1"
__version__ = "0.0.1"
__description__ = "plain C-core, ctypes-interface, clean code approach"
__requirements__ = []
__externalrequirements__ = ["gcc"]
__interpreters__ = ["python3"]
__parallel__ = False
__license__ = "GPLv2"
__authors__ = [
    "Sebastian M. Ernst <ernst@pleiszenburg.de>",
]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import ctypes
import os
import sysconfig

from ._base import UniverseBase

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Universe(UniverseBase):
    __doc__ = __description__
    _LIB = '_libcc1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cdtype = None
        self._step_stage1_c = None
        self._univ_free = None
        self._univ_ptr = None
        self._univ = None

    def start_kernel(self):
        self._cdtype = getattr(
            ctypes,
            f"c_{dict(float32 = 'float', float64 = 'double')[self._dtype]:s}",
        )
        fields = ["rx", "ry", "rz", "ax", "ay", "az", "m"]

        class Mass(ctypes.Structure):
            _fields_ = [
                (field, self._cdtype) for field in fields
            ]

        lib = ctypes.cdll.LoadLibrary(
            os.path.join(os.path.dirname(__file__), self._LIB, f"lib.{sysconfig.get_config_var('SOABI')}.so")
        )
        suffix = dict(float32 = 'f4', float64 = 'f8')[self._dtype]

        univ_alloc = getattr(lib, f'univ_alloc_{suffix:s}')
        univ_alloc.argtypes = (ctypes.POINTER(ctypes.POINTER(Mass)), ctypes.c_size_t)

        self._univ_free = getattr(lib, f'univ_free_{suffix:s}')
        self._univ_free.argtypes = (ctypes.POINTER(ctypes.POINTER(Mass)), ctypes.c_size_t)

        self._step_stage1_c = getattr(lib, f'univ_step_stage1_{suffix:s}')
        self._step_stage1_c.argtypes = (ctypes.POINTER(Mass), self._cdtype, ctypes.c_size_t)

        self._univ_ptr = ctypes.POINTER(Mass)()
        univ_alloc(ctypes.pointer(self._univ_ptr), ctypes.c_size_t(len(self)))

        self._univ = ctypes.cast(self._univ_ptr, ctypes.POINTER(Mass * len(self))).contents
        for idx, pm in enumerate(self._masses):
            self._univ[idx].m = pm.m

    def stop_kernel(self):
        self._univ_free(ctypes.pointer(self._univ_ptr))

    def push_stage1(self):
        for idx, pm in enumerate(self._masses):
            (
                self._univ[idx].rx,
                self._univ[idx].ry,
                self._univ[idx].rz,
            ) = pm.r

    def step_stage1(self):
        self._step_stage1_c(self._univ_ptr, self._cdtype(self._G), ctypes.c_size_t(len(self)))

    def pull_stage1(self):
        for idx, pm in enumerate(self._masses):
            pm.a[:] = [
                self._univ[idx].ax,
                self._univ[idx].ay,
                self._univ[idx].az,
            ]
