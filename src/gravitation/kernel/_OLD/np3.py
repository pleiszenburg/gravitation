# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/np3.py: Kernel

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

__longname__ = "numpy-backend [parallel] (3)"
__version__ = "0.0.1"
__description__ = "numpy backend, parallel, multiprocessing-IPC"
__requirements__ = ["numpy"]
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

import multiprocessing as mp

import numpy as np

from ._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def create_worker_context(m, G, SIM_DIM, MASS_LEN, DTYPE):
    # Store context information in global namespace of worker
    global context
    context = {
        "SIM_DIM": SIM_DIM,
        "MASS_LEN": MASS_LEN,
        # Allocate memory: Object parameters
        "mass_a_array": np.zeros((MASS_LEN, SIM_DIM), dtype=DTYPE),
        "mass_m_array": m,
        # Allocate memory: Temporary variables
        "relative_r": np.zeros((MASS_LEN - 1, SIM_DIM), dtype=DTYPE),
        "distance_sq": np.zeros((MASS_LEN - 1,), dtype=DTYPE),
        "distance_sqv": np.zeros((MASS_LEN - 1, SIM_DIM), dtype=DTYPE),
        "distance_inv": np.zeros((MASS_LEN - 1,), dtype=DTYPE),
        "a_factor": np.zeros((MASS_LEN - 1,), dtype=DTYPE),
        "a1": np.zeros((MASS_LEN - 1,), dtype=DTYPE),
        "a1r": np.zeros((MASS_LEN - 1, SIM_DIM), dtype=DTYPE),
        "a1v": np.zeros((SIM_DIM,), dtype=DTYPE),
        "a2": np.zeros((MASS_LEN - 1,), dtype=DTYPE),
        "a2r": np.zeros((MASS_LEN - 1, SIM_DIM), dtype=DTYPE),
        "G": G,
    }


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class universe(universe_base):
    def start_kernel(self):
        self.DTYPE = self._dtype

        # Get const values
        self.MASS_LEN = len(self)
        self.SIM_DIM = len(self._mass_list[0]._r)
        self.CPU_LEN = self._threads

        # Allocate memory in main process
        self.mass_r_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE)
        mass_m_array = np.zeros((self.MASS_LEN,), dtype=self.DTYPE)

        # Copy const data into Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            mass_m_array[pm_index] = pm._m

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

        # Init multiprocessing pool
        self.cpu_pool = mp.Pool(
            processes=self.CPU_LEN,
            initializer=create_worker_context,
            initargs=(
                mass_m_array,
                self._G,
                self.SIM_DIM,
                self.MASS_LEN,
                self.DTYPE,
            ),
        )

    @staticmethod
    def update_pair(i, k, mass_r_array):
        np.subtract(
            mass_r_array[i, :], mass_r_array[i + 1 :, :], out=context["relative_r"][:k]
        )
        np.multiply(
            context["relative_r"][:k],
            context["relative_r"][:k],
            out=context["distance_sqv"][:k],
        )
        np.add.reduce(
            context["distance_sqv"][:k], axis=1, out=context["distance_sq"][:k]
        )
        np.sqrt(context["distance_sq"][:k], out=context["distance_inv"][:k])
        np.divide(1.0, context["distance_inv"][:k], out=context["distance_inv"][:k])
        np.multiply(
            context["relative_r"][:k],
            context["distance_inv"][:k].reshape(k, 1),
            out=context["relative_r"][:k],
        )
        np.divide(context["G"], context["distance_sq"][:k], out=context["a_factor"][:k])
        np.multiply(
            context["a_factor"][:k],
            context["mass_m_array"][i + 1 :],
            out=context["a1"][:k],
        )
        np.multiply(
            context["a_factor"][:k], context["mass_m_array"][i], out=context["a2"][:k]
        )
        np.multiply(
            context["relative_r"][:k],
            context["a1"][:k].reshape(k, 1),
            out=context["a1r"][:k],
        )
        np.add.reduce(context["a1r"][:k], axis=0, out=context["a1v"])
        np.subtract(
            context["mass_a_array"][i, :],
            context["a1v"],
            out=context["mass_a_array"][i, :],
        )
        np.multiply(
            context["relative_r"][:k],
            context["a2"][:k].reshape(k, 1),
            out=context["a2r"][:k],
        )
        np.add(
            context["mass_a_array"][i + 1 :, :],
            context["a2r"][:k],
            out=context["mass_a_array"][i + 1 :, :],
        )

    @staticmethod
    def step_stage1_batch(i_start, i_end, mass_len, mass_r_array):
        # Zero-out memory
        context["mass_a_array"][:, :] = 0.0
        # Run "pair" calculation: One object against vector of objects per iteration
        for row in range(i_start, i_end):
            universe.update_pair(
                row, mass_len - 1 - row, mass_r_array
            )  # max for temp arrays
        # Return accumulated acceleration
        return context["mass_a_array"]

    def step_stage1(self):
        # Copy dynamic data to Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_r_array[pm_index, :] = pm._r[:]
        # Run "pair" calculation: One object against vector of objects per iteration
        pool_results = [
            self.cpu_pool.apply_async(
                universe.step_stage1_batch,
                args=(
                    i_start,
                    i_end,
                    self.MASS_LEN,
                    self.mass_r_array,
                ),
            )
            for i_start, i_end in self.index_pool
        ]
        result_batches = [result.get() for result in pool_results]
        # Reduce batches
        for batch in result_batches[1:]:
            np.add(result_batches[0], batch, out=result_batches[0])
        # Push dynamic data back to Python infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            pm._a[:] = result_batches[0][pm_index, :]
