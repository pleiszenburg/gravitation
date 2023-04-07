# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/ne1.py: Kernel

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

__longname__ = "numexpr-backend (1)"
__version__ = "0.0.1"
__description__ = "pure numexpr backend, numpy only for holding data"
__requirements__ = ["numexpr", "numpy"]
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

import numpy as np
import numexpr as ne

from ._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class universe(universe_base):
    def start_kernel(self):
        self.DTYPE = self._dtype
        self._G = getattr(np, self.DTYPE)(self._G)
        ne.set_num_threads(ne.detect_number_of_cores())
        # Get const values
        self.MASS_LEN = len(self)
        self.SIM_DIM = len(self._mass_list[0]._r)
        # Allocate memory: Object parameters
        self.mass_r_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE)
        self.mass_a_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE)
        self.mass_m_array = np.zeros((self.MASS_LEN,), dtype=self.DTYPE)
        # Copy const data into Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_m_array[pm_index] = pm._m
        # Allocate memory: Temporary variables
        self.relative_r = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE)
        self.distance_sq = np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.distance_sqv = np.zeros(
            (self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE
        )
        self.distance_inv = np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.a_factor = np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.a1 = np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.a1r = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE)
        self.a1v = np.zeros((self.SIM_DIM,), dtype=self.DTYPE)
        self.a2 = np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE)
        self.a2r = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE)

    def update_pair(self, i, k):
        # np.subtract(self.mass_r_array[i,:], self.mass_r_array[i+1:,:], out = self.relative_r[:k])
        ne.evaluate(
            "a-b",
            local_dict={
                "a": self.mass_r_array[i, :],
                "b": self.mass_r_array[i + 1 :, :],
            },
            out=self.relative_r[:k],
        )

        # np.multiply(self.relative_r[:k], self.relative_r[:k], out = self.distance_sqv[:k])
        # np.add.reduce(self.distance_sqv[:k], axis = 1, out = self.distance_sq[:k])
        ne.evaluate(
            "sum(a**2, axis = 1)",
            local_dict={"a": self.relative_r[:k]},
            out=self.distance_sq[:k],
        )

        # np.sqrt(self.distance_sq[:k], out = self.distance_inv[:k])
        # np.divide(1.0, self.distance_inv[:k], out = self.distance_inv[:k])
        ne.evaluate(
            "a/sqrt(b)",
            local_dict={"a": getattr(np, self.DTYPE)(1.0), "b": self.distance_sq[:k]},
            out=self.distance_inv[:k],
        )

        # np.multiply(self.relative_r[:k], self.distance_inv[:k].reshape(k, 1), out = self.relative_r[:k])
        ne.evaluate(
            "a*b",
            local_dict={
                "a": self.relative_r[:k],
                "b": self.distance_inv[:k].reshape(k, 1),
            },
            out=self.relative_r[:k],
        )

        # np.divide(self._G, self.distance_sq[:k], out = self.a_factor[:k])
        ne.evaluate(
            "a/b",
            local_dict={"a": self._G, "b": self.distance_sq[:k]},
            out=self.a_factor[:k],
        )

        # np.multiply(self.a_factor[:k], self.mass_m_array[i+1:], out = self.a1[:k])
        # np.multiply(self.relative_r[:k], self.a1[:k].reshape(k, 1), out = self.a1r[:k])
        # np.sum(self.a1r[:k], axis = 0, out = self.a1v)
        ne.evaluate(
            "sum(r*f*m, axis = 0)",
            local_dict={
                "f": self.a_factor[:k].reshape(k, 1),
                "m": self.mass_m_array[i + 1 :].reshape(k, 1),
                "r": self.relative_r[:k],
            },
            out=self.a1v,
        )

        # np.subtract(self.mass_a_array[i,:], self.a1v, out = self.mass_a_array[i,:])
        ne.evaluate(
            "a-b",
            local_dict={"a": self.mass_a_array[i, :], "b": self.a1v},
            out=self.mass_a_array[i, :],
        )

        # np.multiply(self.a_factor[:k], self.mass_m_array[i], out = self.a2[:k])
        # np.multiply(self.relative_r[:k], self.a2[:k].reshape(k, 1), out = self.a2r[:k])
        # np.add(self.mass_a_array[i+1:,:], self.a2r[:k], out = self.mass_a_array[i+1:,:])
        ne.evaluate(
            "a+(f*m*r)",
            local_dict={
                "a": self.mass_a_array[i + 1 :, :],
                "f": self.a_factor[:k].reshape(k, 1),
                "m": self.mass_m_array[i],
                "r": self.relative_r[:k],
            },
            out=self.mass_a_array[i + 1 :, :],
        )

    def step_stage1(self):
        # Zero out variables
        self.mass_a_array[:, :] = 0.0
        # Copy dynamic data to Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_r_array[pm_index, :] = pm._r[:]
        # Run "pair" calculation: One object against vector of objects per iteration
        for row in range(0, self.MASS_LEN - 1):
            self.update_pair(row, self.MASS_LEN - 1 - row)  # max for temp arrays
        # Push dynamic data back to Python infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            pm._a[:] = self.mass_a_array[pm_index, :]
