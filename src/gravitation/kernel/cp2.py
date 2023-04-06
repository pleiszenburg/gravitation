# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/cp2.py: Kernel

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

__longname__ = "cupy-backend (2)"
__version__ = "0.0.1"
__description__ = "cupy backend with element-wise cupy-kernel"
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

        # cp.cuda.Device(0).use()
        # # https://github.com/chainer/chainer/issues/3467#issuecomment-333494909
        # memory_pool = cp.cuda.MemoryPool()
        # cp.cuda.set_allocator(memory_pool.malloc)
        # pinned_memory_pool = cp.cuda.PinnedMemoryPool()
        # cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

        # Create cupy kernel for pair update
        self.update_pair_kernel = cp.ElementwiseKernel(
            in_params=(
                "{dtype:s} r1x, {dtype:s} r1y, {dtype:s} r1z, {dtype:s} m1, "
                "{dtype:s} r2x, {dtype:s} r2y, {dtype:s} r2z, {dtype:s} m2"
            ).format(
                dtype=self.DTYPE,
            ),
            out_params=(
                "{dtype:s} a1x, {dtype:s} a1y, {dtype:s} a1z, "
                "{dtype:s} a2x, {dtype:s} a2y, {dtype:s} a2z"
            ).format(
                dtype=self.DTYPE,
            ),
            operation="""
				{dtype:s} relative_rx = r1x - r2x;
				{dtype:s} relative_ry = r1y - r2y;
				{dtype:s} relative_rz = r1z - r2z;
				{dtype:s} distance_sq = (relative_rx * relative_rx) + (relative_ry * relative_ry) + (relative_rz * relative_rz);
				{dtype:s} distance_inv = 1.0 / sqrt(distance_sq);
				relative_rx *= distance_inv;
				relative_ry *= distance_inv;
				relative_rz *= distance_inv;
				{dtype:s} a_factor = {G:e} / distance_sq;
				{dtype:s} a1 = a_factor * m2;
				{dtype:s} a2 = a_factor * m1;
				a1x = relative_rx * a1;
				a1y = relative_ry * a1;
				a1z = relative_rz * a1;
				a2x = relative_rx * a2;
				a2y = relative_ry * a2;
				a2z = relative_rz * a2;
			""".format(
                dtype={"float32": "float", "float64": "double"}[self.DTYPE],
                G=self._G,
            ),
            name="update_pair_kernel",
        )
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
        self.a1r = cp.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE)
        self.a1v = cp.zeros((self.SIM_DIM,), dtype=self.DTYPE)
        self.a2r = cp.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype=self.DTYPE)

    def update_pair(self, i, k):
        (
            self.a1r[:k, 0],
            self.a1r[:k, 1],
            self.a1r[:k, 2],
            self.a2r[:k, 0],
            self.a2r[:k, 1],
            self.a2r[:k, 2],
        ) = self.update_pair_kernel(
            self.mass_r_arrayg[i, 0],
            self.mass_r_arrayg[i, 1],
            self.mass_r_arrayg[i, 2],
            self.mass_m_array[i],
            self.mass_r_arrayg[i + 1 :, 0],
            self.mass_r_arrayg[i + 1 :, 1],
            self.mass_r_arrayg[i + 1 :, 2],
            self.mass_m_array[i + 1 :],
        )
        cp.sum(self.a1r[:k], axis=0, out=self.a1v)
        cp.subtract(self.mass_a_arrayg[i, :], self.a1v, out=self.mass_a_arrayg[i, :])
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
