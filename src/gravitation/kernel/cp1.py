# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/cp1.py: Kernel

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

__longname__ = "cupy-backend (1)"
__version__ = "0.0.1"
__description__ = "numpy-compatible cupy backend"
__requirements__ = ["cupy", "numpy"]
__externalrequirements__ = ["cuda"]
__interpreters__ = ["python3"]
__parallel__ = False
__license__ = "GPLv2"
__authors__ = [
    "Sebastian M. Ernst <ernst@pleiszenburg.de>",
]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import cupy as cp

from ._base_ import universe_base

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
        self.mass_r_arrayc = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE)
        self.mass_r_arrayg = cp.zeros((self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE)
        self.mass_a_arrayc = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE)
        self.mass_a_arrayg = cp.zeros((self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE)
        self.mass_m_array = cp.zeros((self.MASS_LEN,), dtype=self.DTYPE)
        # Copy const data into Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_m_array[pm_index] = pm._m
        # Allocate memory: Temporary variables
        self.relative_r = cp.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE)
        self.distance_sq = cp.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.distance_sqv = cp.zeros(
            (self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE
        )
        self.distance_inv = cp.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.a_factor = cp.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.a1 = cp.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.a1r = cp.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE)
        self.a1v = cp.zeros((self.SIM_DIM,), dtype=self.DTYPE)
        self.a2 = cp.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.a2r = cp.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE)

    def update_pair(self, i, k):
        cp.subtract(
            self.mass_r_arrayg[i, :],
            self.mass_r_arrayg[i + 1 :, :],
            out=self.relative_r[:k],
        )
        cp.multiply(self.relative_r[:k], self.relative_r[:k], out=self.distance_sqv[:k])
        # np.add.reduce(self.distance_sqv[:k], axis = 1, out = self.distance_sq[:k])
        cp.sum(self.distance_sqv[:k], axis=1, out=self.distance_sq[:k])
        cp.sqrt(self.distance_sq[:k], out=self.distance_inv[:k])
        cp.divide(1.0, self.distance_inv[:k], out=self.distance_inv[:k])
        cp.multiply(
            self.relative_r[:k],
            self.distance_inv[:k].reshape(k, 1),
            out=self.relative_r[:k],
        )
        cp.divide(self._G, self.distance_sq[:k], out=self.a_factor[:k])
        cp.multiply(self.a_factor[:k], self.mass_m_array[i + 1 :], out=self.a1[:k])
        cp.multiply(self.a_factor[:k], self.mass_m_array[i], out=self.a2[:k])
        cp.multiply(self.relative_r[:k], self.a1[:k].reshape(k, 1), out=self.a1r[:k])
        # np.add.reduce(self.a1r[:k], axis = 0, out = self.a1v)
        cp.sum(self.a1r[:k], axis=0, out=self.a1v)
        cp.subtract(self.mass_a_arrayg[i, :], self.a1v, out=self.mass_a_arrayg[i, :])
        cp.multiply(self.relative_r[:k], self.a2[:k].reshape(k, 1), out=self.a2r[:k])
        cp.add(
            self.mass_a_arrayg[i + 1 :, :],
            self.a2r[:k],
            out=self.mass_a_arrayg[i + 1 :, :],
        )

    def step_stage1(self):
        # Zero out variables
        self.mass_a_arrayg[:, :] = 0.0
        # Copy dynamic data to Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_r_arrayc[pm_index, :] = pm._r[:]
        # Push data to graphics card
        self.mass_r_arrayg[:, :] = cp.asarray(self.mass_r_arrayc, dtype=self.DTYPE)
        # Run "pair" calculation: One object against vector of objects per iteration
        for row in range(0, self.MASS_LEN - 1):
            self.update_pair(row, self.MASS_LEN - 1 - row)  # max for temp arrays
        # Fetch data from graphics card
        self.mass_a_arrayc[:, :] = cp.asnumpy(self.mass_a_arrayg)
        # Push dynamic data back to Python infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            pm._a[:] = self.mass_a_arrayc[pm_index, :]
