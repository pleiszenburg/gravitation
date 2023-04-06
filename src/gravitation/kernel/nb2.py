# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/nb2.py: Kernel

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

__longname__ = "numba-numpy-backend (2)"
__version__ = "0.0.1"
__description__ = "numba guvectorize-kernel, stage 2 numpy"
__requirements__ = ["numba", "numpy"]
__externalrequirements__ = []
__interpreters__ = ["python3"]
__parallel__ = False
__license__ = "GPLv2"
__authors__ = [
    "Sebastian M. Ernst <ernst@pleiszenburg.de>",
]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numba

import numpy as np

from ._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@numba.guvectorize(
    [
        numba.void(
            numba.float32[:],
            numba.float32[:],
            numba.float32,
            numba.float32,
            numba.float32,
            numba.float32[:],
            numba.float32[:],
        )
    ],
    "(n),(n),(),(),()->(n),(n)",
    nopython=True,
    target="cpu",  # cuda
)
def update_pair_jit(r1, r2, m1, m2, G, a1, a2):
    relative_r = r1 - r2
    distance_sq = np.sum(relative_r**2)
    distance_inv = 1.0 / np.sqrt(distance_sq)
    relative_r *= distance_inv
    a_factor = G / distance_sq
    a1f = a_factor * m2
    a2f = a_factor * m1
    a1 -= relative_r * a1f
    a2 += relative_r * a2f


@numba.jit()  # nopython = True
def step_stage1_jit(
    mass_r_array,
    mass_a_array,
    mass_m_array,
    MASS_LEN,
    G,
):
    mass_a_array[:, :] = 0.0
    for row in range(0, MASS_LEN - 1):
        update_pair_jit(
            mass_r_array[row, :],
            mass_r_array[row + 1 :, :],
            mass_m_array[row],
            mass_m_array[row + 1 :],
            G,
            mass_a_array[row, :],
            mass_a_array[row + 1 :, :],
        )


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class universe(universe_base):
    def start_kernel(self):
        self.DTYPE = self._dtype
        # Get const values
        self.MASS_LEN = len(self)
        self.SIM_DIM = len(self._mass_list[0]._r)
        # Allocate memory: Object parameters
        self.mass_r_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="C"
        )
        self.mass_v_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="C"
        )
        self.mass_a_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="C"
        )
        self.mass_m_array = np.zeros((self.MASS_LEN,), dtype=self.DTYPE, order="C")
        # Copy const data into Numpy infrastructure and link mass objects to Numpy views
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_m_array[pm_index] = pm._m
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_r_array[pm_index, :] = pm._r[:]
            pm._r = self.mass_r_array[pm_index, :]
            self.mass_v_array[pm_index, :] = pm._v[:]
            pm._v = self.mass_v_array[pm_index, :]
            pm._a = self.mass_a_array[pm_index, :]
        # Temp
        self.mass_vt_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE)

    def step_stage1(self):
        step_stage1_jit(
            self.mass_r_array,
            self.mass_a_array,
            self.mass_m_array,
            self.MASS_LEN,
            self._G,
        )

    def step_stage2(self):
        np.multiply(self.mass_a_array, self._T, out=self.mass_a_array)
        np.add(self.mass_v_array, self.mass_a_array, out=self.mass_v_array)
        np.multiply(self.mass_v_array, self._T, out=self.mass_vt_array)
        np.add(self.mass_r_array, self.mass_vt_array, out=self.mass_r_array)
        self.mass_a_array[:, :] = 0.0
