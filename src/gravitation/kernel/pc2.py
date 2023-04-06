# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/pc2.py: Kernel

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

__longname__ = "pycuda-backend (2)"
__version__ = "0.0.1"
__description__ = "pycuda, O(N*(N-1)) complexity"
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
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

SM = """
__global__ void update_all(
	{dtype:s} *rx, {dtype:s} *ry, {dtype:s} *rz,
	{dtype:s} *ax, {dtype:s} *ay, {dtype:s} *az,
	{dtype:s} *m
	)
{{
	{itype:s}_t j = threadIdx.x + blockIdx.x * blockDim.x;
	if(j >= {MASS_LEN:d}) return;
	{dtype:s} relative_rx, relative_ry, relative_rz, distance_sq, a_factor;
	{dtype:s} atx = 0.0;
	{dtype:s} aty = 0.0;
	{dtype:s} atz = 0.0;
	{dtype:s} rtx = rx[j];
	{dtype:s} rty = ry[j];
	{dtype:s} rtz = rz[j];
	for({itype:s}_t i = 0; i != {MASS_LEN:d}; i++)
	{{
		if(i == j) continue;
		relative_rx = rx[i] - rtx;
		relative_ry = ry[i] - rty;
		relative_rz = rz[i] - rtz;
		distance_sq = (relative_rx * relative_rx) + (relative_ry * relative_ry) + (relative_rz * relative_rz);
		a_factor = m[i] * {rsqrt:s}(distance_sq) / distance_sq;
		atx += relative_rx * a_factor;
		aty += relative_ry * a_factor;
		atz += relative_rz * a_factor;
	}}
	ax[j] = atx * {G:e};
	ay[j] = aty * {G:e};
	az[j] = atz * {G:e};
}}
"""

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
            SM.format(
                dtype={"float32": "float", "float64": "double"}[self.DTYPE],
                itype=self.ITYPE,
                rsqrt={"float32": "rsqrtf", "float64": "rsqrt"}[self.DTYPE],
                G=self._G,
                MASS_LEN=self.MASS_LEN,
            )
        )
        self.sm_update_all = self.sm.get_function("update_all")
        # Compute cuda threads and blocks
        self.threads_per_block = (
            256  # 512 # http://recurial.com/wp-content/uploads/2016/02/p1-vollmer.pdf
        )
        self.blocks = self.MASS_LEN // self.threads_per_block
        if (self.MASS_LEN % self.threads_per_block) != 0:
            self.blocks += 1

    def step_stage1(self):
        # Copy data to GPU memory
        cuda.memcpy_htod(self.mass_rx_array_g, self.mass_r_array[:, 0])
        cuda.memcpy_htod(self.mass_ry_array_g, self.mass_r_array[:, 1])
        cuda.memcpy_htod(self.mass_rz_array_g, self.mass_r_array[:, 2])
        # Run "pair" calculation: One object against vector of objects per iteration
        self.sm_update_all(
            self.mass_rx_array_g,
            self.mass_ry_array_g,
            self.mass_rz_array_g,
            self.mass_ax_array_g,
            self.mass_ay_array_g,
            self.mass_az_array_g,
            self.mass_m_array_g,
            block=(self.threads_per_block, 1, 1),
            grid=(self.blocks, 1),
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
