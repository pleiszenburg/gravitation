# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/pc1/kernel.py: Kernel

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
__version__ = "0.0.2"
__description__ = "pycuda, attempt with O(N*(N-1)/2) complexity [MEMORY RACE!]"  # TODO
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

from math import ceil

import numpy as np

import pycuda.autoinit  # pylint: disable=unused-import
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from ...lib.base import UniverseBase
from ...lib.const import DIMS
from ...lib.debug import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

SRC = """
__global__ void update_pair(
    {itype:s}_t i, {itype:s}_t len,
    {dtype:s} *rx, {dtype:s} *ry, {dtype:s} *rz,
    {dtype:s} *ax, {dtype:s} *ay, {dtype:s} *az,
    {dtype:s} *m
    )
{{

    {itype:s}_t j = i + 1 + (threadIdx.x + blockIdx.x * blockDim.x);

    if(j >= len){{
        return;
    }}

    {dtype:s} dx = rx[i] - rx[j];
    {dtype:s} dy = ry[i] - ry[j];
    {dtype:s} dz = rz[i] - rz[j];

    {dtype:s} dxyz = (dx * dx) + (dy * dy) + (dz * dz);

    {dtype:s} dxyzg = {G:e} / dxyz;

    {dtype:s} aj = dxyzg * m[i];
    {dtype:s} ai = dxyzg * m[j];

    dxyz = {rsqrt:s}(dxyz);

    dx *= dxyz;
    dy *= dxyz;
    dz *= dxyz;

    ax[j] += aj * dx;
    ay[j] += aj * dy;
    az[j] += aj * dz;

    ax[i] -= ai * dx;  // BUG: RACE CONDITION!
    ay[i] -= ai * dy;  // BUG: RACE CONDITION!
    az[i] -= ai * dz;  // BUG: RACE CONDITION!

}}
"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Universe(UniverseBase):
    __doc__ = __description__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dtype_size = getattr(np, self._dtype)(0).itemsize
        self._itype = 'int32'

        self._mem_size = None
        self._len = None

        self._r = None
        self._a = None
        self._m = None

        self._rx_gpu = None
        self._ry_gpu = None
        self._rz_gpu = None
        self._ax_gpu = None
        self._ay_gpu = None
        self._az_gpu = None
        self._m_gpu = None

        self._sm = None
        self._update_pair = None
        self._index_range = None

    def start_kernel(self):

        self._mem_size = self._dtype_size * len(self)
        self._len = getattr(np, self._itype)(len(self))

        # Allocate memory: Object parameters
        self._r = np.zeros((DIMS, len(self)), dtype=self._dtype)
        self._a = np.zeros((DIMS, len(self)), dtype=self._dtype)
        self._m = np.zeros((len(self),), dtype=self._dtype)

        # Allocate memory: pycuda
        self._rx_gpu = cuda.mem_alloc(self._r[0, :].nbytes)  # pylint: disable=no-member
        self._ry_gpu = cuda.mem_alloc(self._r[1, :].nbytes)  # pylint: disable=no-member
        self._rz_gpu = cuda.mem_alloc(self._r[2, :].nbytes)  # pylint: disable=no-member
        self._ax_gpu = cuda.mem_alloc(self._a[0, :].nbytes)  # pylint: disable=no-member
        self._ay_gpu = cuda.mem_alloc(self._a[1, :].nbytes)  # pylint: disable=no-member
        self._az_gpu = cuda.mem_alloc(self._a[2, :].nbytes)  # pylint: disable=no-member
        self._m_gpu = cuda.mem_alloc(self._m.nbytes)  # pylint: disable=no-member

        # Copy const data into Numpy infrastructure
        for idx, pm in enumerate(self._masses):
            self._m[idx] = pm.m

        # Copy const data into GPU memory
        cuda.memcpy_htod(self._m_gpu, self._m)  # pylint: disable=no-member

        # Create cuda kernel for pair update
        self._sm = SourceModule(SRC.format(
                dtype={"float32": "float", "float64": "double"}[self._dtype],
                itype=self._itype,
                rsqrt={"float32": "rsqrtf", "float64": "rsqrt"}[self._dtype],
                G=self._G,
            )
        )
        self._update_pair = self._sm.get_function("update_pair")

        # Compute cuda threads and blocks
        threads_per_block = 256  # 512
        self._index_range = []
        for row in range(0, len(self) - 1):
            threads_total = len(self) - 1 - row
            blocks = ceil(threads_total / threads_per_block)
            self._index_range.append(
                (getattr(np, self._itype)(row), threads_per_block, blocks)
            )

    def push_stage1(self):

        for idx, pm in enumerate(self._masses):
            self._r[:, idx] = pm.r[:]

        cuda.memcpy_htod(self._rx_gpu, self._r[0, :])  # pylint: disable=no-member
        cuda.memcpy_htod(self._ry_gpu, self._r[1, :])  # pylint: disable=no-member
        cuda.memcpy_htod(self._rz_gpu, self._r[2, :])  # pylint: disable=no-member

    def step_stage1(self):

        cuda.memset_d8(self._ax_gpu, 0, self._mem_size)  # pylint: disable=no-member
        cuda.memset_d8(self._ay_gpu, 0, self._mem_size)  # pylint: disable=no-member
        cuda.memset_d8(self._az_gpu, 0, self._mem_size)  # pylint: disable=no-member

        # Run "pair" calculation: One object against vector of objects per iteration
        for row_np, threads_per_block, blocks in self._index_range:
            self._update_pair(
                row_np,
                self._len,
                self._rx_gpu,
                self._ry_gpu,
                self._rz_gpu,
                self._ax_gpu,
                self._ay_gpu,
                self._az_gpu,
                self._m_gpu,
                block=(threads_per_block, 1, 1),
                grid=(blocks, 1),
            )

    def pull_stage1(self):

        cuda.memcpy_dtoh(self._a[0, :], self._ax_gpu)  # pylint: disable=no-member
        cuda.memcpy_dtoh(self._a[1, :], self._ay_gpu)  # pylint: disable=no-member
        cuda.memcpy_dtoh(self._a[2, :], self._az_gpu)  # pylint: disable=no-member

        for idx, pm in enumerate(self._masses):
            pm.a[:] = [float(v) for v in self._a[:, idx]]
