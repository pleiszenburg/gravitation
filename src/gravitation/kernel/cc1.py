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
__version__ = "0.1.0"
__description__ = "plain C-core, ctypes-interface"
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
        self._univ = None

    def start_kernel(self):
        self._cdtype = getattr(
            ctypes,
            f"c_{dict(float32 = 'float', float64 = 'double')[self._dtype]:s}",
        )
        array_fields = ["x", "y", "z", "ax", "ay", "az", "m"]
        array_type = self._cdtype * len(self)

        class Univ(ctypes.Structure):
            _fields_ = [
                (field, ctypes.POINTER(array_type)) for field in array_fields
            ] + [
                ("g", self._cdtype),
                ("n", ctypes.c_size_t),
            ]

        self._step_stage1_c = ctypes.cdll.LoadLibrary(
            os.path.join(os.path.dirname(__file__), self._LIB, f"lib.{sysconfig.get_config_var('SOABI')}.so")
        ).univ_step_stage1

        self._step_stage1_c.argtypes = (ctypes.POINTER(Univ),)
        self._univ = Univ()
        for field in array_fields:
            getattr(self._univ, field).contents = array_type()
        for idx, pm in enumerate(self._masses):
            self._univ.m.contents[idx] = pm.m
        self._univ.g = self._G
        self._univ.n = len(self)

    def push_stage1(self):
        for idx, pm in enumerate(self._masses):
            (
                self._univ.x.contents[idx],
                self._univ.y.contents[idx],
                self._univ.z.contents[idx],
            ) = pm.r

    def step_stage1(self):
        self._step_stage1_c(self._univ)

    def pull_stage1(self):
        for idx, pm in enumerate(self._masses):
            pm.a[:] = [
                self._univ.ax.contents[idx],
                self._univ.ay.contents[idx],
                self._univ.az.contents[idx],
            ]
