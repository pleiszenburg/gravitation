# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/c1a.py: Kernel

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

__longname__ = "c-backend 1(a)"
__version__ = "0.0.1"
__description__ = "C-core, ctypes-interface"
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

from ._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class universe(universe_base):
    def start_kernel(self):
        self.DTYPE = self._dtype
        self.CDTYPE = getattr(
            ctypes,
            "c_{name:s}".format(
                name={"float32": "float", "float64": "double"}[self.DTYPE]
            ),
        )
        array_fields = ["X", "Y", "Z", "AX", "AY", "AZ", "M"]
        array_type = self.CDTYPE * len(self)

        class univ(ctypes.Structure):
            _fields_ = [
                (field, ctypes.POINTER(array_type)) for field in array_fields
            ] + [
                ("G", self.CDTYPE),
                ("N", ctypes.c_long),
            ]

        self._step_stage1_ = ctypes.cdll.LoadLibrary(
            os.path.join(os.path.dirname(__file__), "_lib1_", "lib.so")
        ).step_stage1
        self._step_stage1_.argtypes = (ctypes.POINTER(univ),)
        self.univ = univ()
        for field in array_fields:
            getattr(self.univ, field).contents = array_type()
        for i, pm in enumerate(self._mass_list):
            self.univ.M.contents[i] = pm._m
        self.univ.G = self._G
        self.univ.N = len(self)

    def step_stage1(self):
        for i, pm in enumerate(self._mass_list):
            (
                self.univ.X.contents[i],
                self.univ.Y.contents[i],
                self.univ.Z.contents[i],
            ) = pm._r
            (
                self.univ.AX.contents[i],
                self.univ.AY.contents[i],
                self.univ.AZ.contents[i],
            ) = (0.0, 0.0, 0.0)
        self._step_stage1_(self.univ)
        for i, pm in enumerate(self._mass_list):
            pm._a[:] = [
                self.univ.AX.contents[i],
                self.univ.AY.contents[i],
                self.univ.AZ.contents[i],
            ]
