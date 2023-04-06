# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/np4.py: Kernel

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

__longname__ = "numpy-backend [parallel] (4)"
__version__ = "0.0.1"
__description__ = "numpy backend, parallel, joblib-IPC"
__requirements__ = ["joblib", "numpy"]
__externalrequirements__ = []
__interpreters__ = ["python3"]
__parallel__ = True
__license__ = "GPLv2"
__authors__ = [
    "Sebastian M. Ernst <ernst@pleiszenburg.de>",
]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import joblib

import numpy as np

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

        # Init multiprocessing pool
        self.CPU_LEN = self._threads
        self.cpu_pool = joblib.Parallel(
            n_jobs=self.CPU_LEN, prefer="processes"  # alternative: 'threads'
        )

        self.data_pool = []

        for _ in range(self.CPU_LEN):
            self.data_pool.append(
                {
                    # Allocate memory: Object parameters
                    "mass_r_array": np.zeros(
                        (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE
                    ),
                    "mass_a_array": np.zeros(
                        (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE
                    ),
                    "mass_m_array": np.zeros((self.MASS_LEN,), dtype=self.DTYPE),
                    # Allocate memory: Temporary variables
                    "relative_r": np.zeros(
                        (self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE
                    ),
                    "distance_sq": np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE),
                    "distance_sqv": np.zeros(
                        (self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE
                    ),
                    "distance_inv": np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE),
                    "a_factor": np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE),
                    "a1": np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE),
                    "a1r": np.zeros(
                        (self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE
                    ),
                    "a1v": np.zeros((self.SIM_DIM,), dtype=self.DTYPE),
                    "a2": np.zeros((self.MASS_LEN - 1,), dtype=self.DTYPE),
                    "a2r": np.zeros(
                        (self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE
                    ),
                    "G": self._G,
                }
            )

        # Copy const data into Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            self.data_pool[0]["mass_m_array"][pm_index] = pm._m
        for data_set in self.data_pool[1:]:
            data_set["mass_m_array"][:] = self.data_pool[0]["mass_m_array"][:]

        # Compute line index tuples for evenly sized batches
        total_pairs = (self.MASS_LEN * (self.MASS_LEN - 1)) // 2
        batch_length = total_pairs // self.CPU_LEN
        self.index_pool = []
        pair_count = 0
        start_line = 0
        for line in range(1, self.MASS_LEN - 1):
            pair_count += self.MASS_LEN - 1 - line
            if pair_count < batch_length:
                continue
            pair_count = 0
            self.index_pool.append((start_line, line))
            start_line = line
        assert len(self.index_pool) in [(self.CPU_LEN - 1), self.CPU_LEN]
        if len(self.index_pool) == self.CPU_LEN - 1:
            self.index_pool.append((start_line, self.MASS_LEN - 1))
        assert self.index_pool[-1][1] == self.MASS_LEN - 1

    @staticmethod
    def update_pair(i, k, data_set):
        np.subtract(
            data_set["mass_r_array"][i, :],
            data_set["mass_r_array"][i + 1 :, :],
            out=data_set["relative_r"][:k],
        )
        np.multiply(
            data_set["relative_r"][:k],
            data_set["relative_r"][:k],
            out=data_set["distance_sqv"][:k],
        )
        np.add.reduce(
            data_set["distance_sqv"][:k], axis=1, out=data_set["distance_sq"][:k]
        )
        np.sqrt(data_set["distance_sq"][:k], out=data_set["distance_inv"][:k])
        np.divide(1.0, data_set["distance_inv"][:k], out=data_set["distance_inv"][:k])
        np.multiply(
            data_set["relative_r"][:k],
            data_set["distance_inv"][:k].reshape(k, 1),
            out=data_set["relative_r"][:k],
        )
        np.divide(
            data_set["G"], data_set["distance_sq"][:k], out=data_set["a_factor"][:k]
        )
        np.multiply(
            data_set["a_factor"][:k],
            data_set["mass_m_array"][i + 1 :],
            out=data_set["a1"][:k],
        )
        np.multiply(
            data_set["a_factor"][:k],
            data_set["mass_m_array"][i],
            out=data_set["a2"][:k],
        )
        np.multiply(
            data_set["relative_r"][:k],
            data_set["a1"][:k].reshape(k, 1),
            out=data_set["a1r"][:k],
        )
        np.add.reduce(data_set["a1r"][:k], axis=0, out=data_set["a1v"])
        np.subtract(
            data_set["mass_a_array"][i, :],
            data_set["a1v"],
            out=data_set["mass_a_array"][i, :],
        )
        np.multiply(
            data_set["relative_r"][:k],
            data_set["a2"][:k].reshape(k, 1),
            out=data_set["a2r"][:k],
        )
        np.add(
            data_set["mass_a_array"][i + 1 :, :],
            data_set["a2r"][:k],
            out=data_set["mass_a_array"][i + 1 :, :],
        )

    @staticmethod
    def step_stage1_batch(i_start, i_end, mass_len, data_set):
        # Run "pair" calculation: One object against vector of objects per iteration
        for row in range(i_start, i_end):
            universe.update_pair(
                row, mass_len - 1 - row, data_set
            )  # max for temp arrays
        # Return accumulated acceleration
        return data_set["mass_a_array"]

    def step_stage1(self):
        # Zero out variables
        for data_set in self.data_pool:
            data_set["mass_a_array"][:, :] = 0.0
        # Copy dynamic data to Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            self.data_pool[0]["mass_r_array"][pm_index, :] = pm._r[:]
        for data_set in self.data_pool[1:]:
            data_set["mass_r_array"][:, :] = self.data_pool[0]["mass_r_array"][:, :]
        # Run "pair" calculation: One object against vector of objects per iteration
        result_batches = self.cpu_pool(
            joblib.delayed(universe.step_stage1_batch)(
                i_start, i_end, self.MASS_LEN, data_set
            )
            for (i_start, i_end), data_set in zip(self.index_pool, self.data_pool)
        )
        # Reduce batches
        for batch in result_batches[1:]:
            np.add(result_batches[0], batch, out=result_batches[0])
        # Push dynamic data back to Python infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            pm._a[:] = result_batches[0][pm_index, :]
