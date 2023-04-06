# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/c4b.py: Kernel

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
# KERNEL META
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__longname__ = "c-backend 4(b)"
__version__ = "0.0.1"
__description__ = "C-core, SSE2-intrinsics, openMP-parallel, numpy-ctypes-interface"
__requirements__ = ["numpy"]
__externalrequirements__ = ["gcc"]
__interpreters__ = ["python3"]
__parallel__ = True
__license__ = "GPLv2"
__authors__ = [
    "Sebastian M. Ernst <ernst@pleiszenburg.de>",
]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import ctypes
import os

import numpy as np

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
        # Get const values
        self.MASS_LEN = len(self)
        self.SIM_DIM = len(self._mass_list[0]._r)
        # Manage OpenMP
        os.environ["OMP_NUM_THREADS"] = str(self._threads)
        # Build data structure
        array_fields_r = ["X", "Y", "Z"]
        array_fields_a = ["AX", "AY", "AZ"]
        array_fields_m = ["M"]
        array_fields_amp = ["AXmp", "AYmp", "AZmp"]

        class univ(ctypes.Structure):
            _fields_ = (
                [
                    (field, ctypes.POINTER(self.CDTYPE * self.MASS_LEN))
                    for field in (array_fields_r + array_fields_a + array_fields_m)
                ]
                + [
                    ("G", self.CDTYPE),
                    ("N", ctypes.c_long),
                ]
                + [
                    (
                        field,
                        ctypes.POINTER(self.CDTYPE * (self.MASS_LEN * self._threads)),
                    )
                    for field in array_fields_amp
                ]
                + [
                    ("j_min", ctypes.POINTER(ctypes.c_long)),
                    ("j_max", ctypes.POINTER(ctypes.c_long)),
                    ("seg_len", ctypes.c_long),
                    ("OPENMP_threadsmax", ctypes.c_long),
                ]
            )

        # Attach to library
        lib = ctypes.cdll.LoadLibrary(
            os.path.join(os.path.dirname(__file__), "_lib4_", "lib.so")
        )
        self._step_stage1_segmentation_ = lib.step_stage1_segmentation
        self._step_stage1_segmentation_.argtypes = (ctypes.POINTER(univ),)
        self._step_stage1_ = lib.step_stage1
        self._step_stage1_.argtypes = (ctypes.POINTER(univ),)

        # Allocate memory: Object parameters
        self.mass_r_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="F"
        )
        self.mass_v_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="F"
        )
        self.mass_a_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="F"
        )
        self.mass_amp_array = np.zeros(
            (self.MASS_LEN * self._threads, self.SIM_DIM), dtype=self.DTYPE, order="F"
        )
        self.mass_m_array = np.zeros((self.MASS_LEN,), dtype=self.DTYPE, order="F")

        for np_array in [
            self.mass_r_array,
            self.mass_v_array,
            self.mass_a_array,
            self.mass_amp_array,
            self.mass_m_array,
        ]:
            assert np_array.flags["F_CONTIGUOUS"] == True
            assert np_array.flags["ALIGNED"] == True
            assert np_array.flags["OWNDATA"] == True
            # assert np_array.flags['FARRAY'] == True

        # Copy const data into Numpy infrastructure and link mass objects to Numpy views
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_m_array[pm_index] = pm._m
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_r_array[pm_index, :] = pm._r[:]
            pm._r = self.mass_r_array[pm_index, :]
            self.mass_v_array[pm_index, :] = pm._v[:]
            pm._v = self.mass_v_array[pm_index, :]
            pm._a = self.mass_a_array[pm_index, :]
        # Fill data structure
        self.univ = univ()
        for index, (field_r, field_a) in enumerate(zip(array_fields_r, array_fields_a)):
            setattr(
                self.univ,
                field_r,
                self.mass_r_array[:, index].ctypes.data_as(
                    ctypes.POINTER(self.CDTYPE * self.MASS_LEN)
                ),
            )
            setattr(
                self.univ,
                field_a,
                self.mass_a_array[:, index].ctypes.data_as(
                    ctypes.POINTER(self.CDTYPE * self.MASS_LEN)
                ),
            )
        setattr(
            self.univ,
            array_fields_m[0],
            self.mass_m_array.ctypes.data_as(
                ctypes.POINTER(self.CDTYPE * self.MASS_LEN)
            ),
        )
        for index, field_amp in enumerate(array_fields_amp):
            setattr(
                self.univ,
                field_amp,
                self.mass_amp_array[:, index].ctypes.data_as(
                    ctypes.POINTER(self.CDTYPE * (self.MASS_LEN * self._threads))
                ),
            )
        self.univ.G = self._G
        self.univ.N = len(self._mass_list)
        self._step_stage1_segmentation_(self.univ)

    def step_stage1(self):
        self.mass_a_array[:, :] = 0.0
        self._step_stage1_(self.univ)
