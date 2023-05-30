# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/pc3.py: Kernel

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

__longname__ = "pycuda-backend (3)"
__version__ = "0.0.1"
__description__ = (
    "pycuda, O(N*(N-1)) complexity, optimization experiment - UNSTABLE, BUGS!"  # TODO bodies keep "disappearing"
)
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
	{dtype:s} rjx = rx[j];
	{dtype:s} rjy = ry[j];
	{dtype:s} rjz = rz[j];

	for({itype:s}_t k = 0; k != gridDim.x; k++)
	{{

		__shared__ {dtype:s} rix[{TPB:d}];
		__shared__ {dtype:s} riy[{TPB:d}];
		__shared__ {dtype:s} riz[{TPB:d}];
		__shared__ {dtype:s} mi[{TPB:d}];

		{itype:s}_t kk = k * {TPB:d};
		{itype:s}_t p = kk + threadIdx.x;

		if(p < {MASS_LEN:d})
		{{
			rix[threadIdx.x] = rx[p];
			riy[threadIdx.x] = ry[p];
			riz[threadIdx.x] = rz[p];
			mi[threadIdx.x] = m[p];
		}}

		__syncthreads();

		#pragma unroll
		for({itype:s}_t i = 0; i != {TPB:d}; i++)
		{{
			if((kk + i) == j) continue;
			if((kk + i) >= {MASS_LEN:d}) break;
			relative_rx = rix[i] - rjx;
			relative_ry = riy[i] - rjy;
			relative_rz = riz[i] - rjz;
			distance_sq =
				(relative_rx * relative_rx)
				+ (relative_ry * relative_ry)
				+ (relative_rz * relative_rz);
			a_factor = mi[i] * {rsqrt:s}(distance_sq) / distance_sq;
			atx += relative_rx * a_factor;
			aty += relative_ry * a_factor;
			atz += relative_rz * a_factor;
		}}

		__syncthreads();

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
        # Compute cuda threads and blocks
        self.threads_per_block = (
            128  # 512 # http://recurial.com/wp-content/uploads/2016/02/p1-vollmer.pdf
        )
        self.blocks = self.MASS_LEN // self.threads_per_block
        if (self.MASS_LEN % self.threads_per_block) != 0:
            self.blocks += 1
        # Create cuda kernel for pair update
        self.sm = SourceModule(
            SM.format(
                dtype={"float32": "float", "float64": "double"}[self.DTYPE],
                itype=self.ITYPE,
                rsqrt={"float32": "rsqrtf", "float64": "rsqrt"}[self.DTYPE],
                G=self._G,
                MASS_LEN=self.MASS_LEN,
                TPB=self.threads_per_block,
            )
        )
        self.sm_update_all = self.sm.get_function("update_all")

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
