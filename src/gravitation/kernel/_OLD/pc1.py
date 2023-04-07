# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/pc1.py: Kernel

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

__longname__ = "pycuda-backend (1)"
__version__ = "0.0.1"
__description__ = "pycuda, attempt with O(N*(N-1)/2) complexity"
__requirements__ = ["numpy", "pycuda"]
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

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from ._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class universe(universe_base):
    def start_kernel(self):
        self.DTYPE = self._dtype
        self.DTYPESIZE = getattr(np, self.DTYPE)().itemsize
        self.ITYPE = "int32"
        # Get const values
        self.MASS_LEN = len(self)
        self.SIM_DIM = len(self._mass_list[0]._r)
        self.MEM_LEN = self.DTYPESIZE * self.MASS_LEN
        # Allocate memory: Object parameters
        self.mass_r_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="F"
        )
        self.mass_v_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="F"
        )
        self.mass_vt_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="F"
        )
        self.mass_a_array = np.zeros(
            (self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE, order="F"
        )
        self.mass_m_array = np.zeros((self.MASS_LEN,), dtype=self.DTYPE)
        # Allocate memory: pycuda
        self.mass_rx_array_g = cuda.mem_alloc(self.mass_r_array[:, 0].nbytes)
        self.mass_ry_array_g = cuda.mem_alloc(self.mass_r_array[:, 1].nbytes)
        self.mass_rz_array_g = cuda.mem_alloc(self.mass_r_array[:, 2].nbytes)
        self.mass_ax_array_g = cuda.mem_alloc(self.mass_a_array[:, 0].nbytes)
        self.mass_ay_array_g = cuda.mem_alloc(self.mass_a_array[:, 1].nbytes)
        self.mass_az_array_g = cuda.mem_alloc(self.mass_a_array[:, 2].nbytes)
        self.mass_m_array_g = cuda.mem_alloc(self.mass_m_array.nbytes)
        # Copy const data into Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_m_array[pm_index] = pm._m
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_r_array[pm_index, :] = pm._r[:]
            pm._r = self.mass_r_array[pm_index, :]
            self.mass_v_array[pm_index, :] = pm._v[:]
            pm._v = self.mass_v_array[pm_index, :]
            pm._a = self.mass_a_array[pm_index, :]
        # Copy const data into GPU memory
        cuda.memcpy_htod(self.mass_m_array_g, self.mass_m_array)
        # Create cuda kernel for pair update
        self.sm = SourceModule(
            """
			__global__ void update_pair(
				{itype:s}_t i, {itype:s}_t MASS_LEN,
				{dtype:s} *rx, {dtype:s} *ry, {dtype:s} *rz,
				{dtype:s} *ax, {dtype:s} *ay, {dtype:s} *az,
				{dtype:s} *m
				)
			{{
				{itype:s}_t j = i + 1 + (threadIdx.x + blockIdx.x * blockDim.x);
				if(j >= MASS_LEN) return;
				{dtype:s} relative_rx = rx[i] - rx[j];
				{dtype:s} relative_ry = ry[i] - ry[j];
				{dtype:s} relative_rz = rz[i] - rz[j];
				{dtype:s} distance_sq = (relative_rx * relative_rx) + (relative_ry * relative_ry) + (relative_rz * relative_rz);
				{dtype:s} distance_inv = {rsqrt:s}(distance_sq);
				relative_rx *= distance_inv;
				relative_ry *= distance_inv;
				relative_rz *= distance_inv;
				{dtype:s} a_factor = {G:e} / distance_sq;
				{dtype:s} a1 = a_factor * m[j];
				{dtype:s} a2 = a_factor * m[i];
				ax[i] += relative_rx * a1; // BUG: RACE CONDITION!
				ay[i] += relative_ry * a1; // BUG: RACE CONDITION!
				az[i] += relative_rz * a1; // BUG: RACE CONDITION!
				ax[j] += relative_rx * a2;
				ay[j] += relative_ry * a2;
				az[j] += relative_rz * a2;
			}}
			""".format(
                dtype={"float32": "float", "float64": "double"}[self.DTYPE],
                itype=self.ITYPE,
                rsqrt={"float32": "rsqrtf", "float64": "rsqrt"}[self.DTYPE],
                G=self._G,
            )
        )
        self.sm_update_pair = self.sm.get_function("update_pair")
        # Compute cuda threads and blocks
        self.index_range = []
        threads_per_block = (
            256  # 512 # http://recurial.com/wp-content/uploads/2016/02/p1-vollmer.pdf
        )
        self.MASS_LEN_np = getattr(np, self.ITYPE)(self.MASS_LEN)
        for row in range(0, self.MASS_LEN - 1):
            threads_total = self.MASS_LEN - 1 - row
            blocks = threads_total // threads_per_block
            if (threads_total % threads_per_block) != 0:
                blocks += 1
            self.index_range.append(
                (getattr(np, self.ITYPE)(row), threads_per_block, blocks)
            )

    def step_stage1(self):
        # Copy data to GPU memory
        cuda.memcpy_htod(self.mass_rx_array_g, self.mass_r_array[:, 0])
        cuda.memcpy_htod(self.mass_ry_array_g, self.mass_r_array[:, 1])
        cuda.memcpy_htod(self.mass_rz_array_g, self.mass_r_array[:, 2])
        cuda.memset_d8(self.mass_ax_array_g, 0, self.MEM_LEN)
        cuda.memset_d8(self.mass_ay_array_g, 0, self.MEM_LEN)
        cuda.memset_d8(self.mass_az_array_g, 0, self.MEM_LEN)
        # Run "pair" calculation: One object against vector of objects per iteration
        for row_np, threads_per_block, blocks in self.index_range:
            self.sm_update_pair(
                row_np,
                self.MASS_LEN_np,
                self.mass_rx_array_g,
                self.mass_ry_array_g,
                self.mass_rz_array_g,
                self.mass_ax_array_g,
                self.mass_ay_array_g,
                self.mass_az_array_g,
                self.mass_m_array_g,
                block=(threads_per_block, 1, 1),
                grid=(blocks, 1),
            )
        # Copy data to GPU memory
        cuda.memcpy_dtoh(self.mass_a_array[:, 0], self.mass_ax_array_g)
        cuda.memcpy_dtoh(self.mass_a_array[:, 1], self.mass_ay_array_g)
        cuda.memcpy_dtoh(self.mass_a_array[:, 2], self.mass_az_array_g)

    def step_stage2(self):
        np.multiply(self.mass_a_array, self._T, out=self.mass_a_array)
        np.add(self.mass_v_array, self.mass_a_array, out=self.mass_v_array)
        np.multiply(self.mass_v_array, self._T, out=self.mass_vt_array)
        np.add(self.mass_r_array, self.mass_vt_array, out=self.mass_r_array)
