# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/np_p3/kernel.py: Kernel

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

__longname__ = "numpy-backend (3)"
__version__ = "0.0.1"
__description__ = "numpy backend, no memory allocations, optimal memory layout, parallel processes, shared memory"
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

import gc

import numpy as np

from ...lib.base import UniverseBase
from ...lib.const import Dtype, DEFAULT_DTYPE
from ...lib.block import Block
from ...lib.const import DIMS
from ...lib.shm import Param, ShmPool, WorkerBase
from ...lib.debug import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class UniverseWorker(WorkerBase):
    "Runs in process"

    def __init__(self, *args, length: int = 0, dtype: Dtype = DEFAULT_DTYPE, G: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)

        self._length = length
        self._G = G

        # Allocate memory: Temporary variables
        self._relative_r = np.zeros((DIMS, length - 1), dtype=dtype.name)
        self._distance_sq = np.zeros((length - 1,), dtype=dtype.name)
        self._distance_sqv = np.zeros((DIMS, length - 1), dtype=dtype.name)
        self._distance_inv = np.zeros((length - 1,), dtype=dtype.name)
        self._a_factor = np.zeros((length - 1,), dtype=dtype.name)
        self._a1 = np.zeros((length - 1,), dtype=dtype.name)
        self._a1r = np.zeros((DIMS, length - 1), dtype=dtype.name)
        self._a1v = np.zeros((DIMS,), dtype=dtype.name)
        self._a2 = np.zeros((length - 1,), dtype=dtype.name)
        self._a2r = np.zeros((DIMS, length - 1), dtype=dtype.name)

        gc.disable()

    def _update_pair(self, i: int, k: int):
        np.subtract(
            self['r'][:, i, None],
            self['r'][:, i + 1 :],
            out=self._relative_r[:, :k],
        )
        np.multiply(
            self._relative_r[:, :k],
            self._relative_r[:, :k],
            out=self._distance_sqv[:, :k],
        )
        np.add.reduce(self._distance_sqv[:, :k], axis=0, out=self._distance_sq[:k])
        np.sqrt(self._distance_sq[:k], out=self._distance_inv[:k])
        np.divide(1.0, self._distance_inv[:k], out=self._distance_inv[:k])
        np.multiply(
            self._relative_r[:, :k],
            self._distance_inv[None, :k],
            out=self._relative_r[:, :k],
        )
        np.divide(self._G, self._distance_sq[:k], out=self._a_factor[:k])
        np.multiply(self._a_factor[:k], self['m'][i + 1 :], out=self._a1[:k])
        np.multiply(self._a_factor[:k], self['m'][i], out=self._a2[:k])
        np.multiply(self._relative_r[:, :k], self._a1[None, :k], out=self._a1r[:, :k])
        np.add.reduce(self._a1r[:, :k], axis=1, out=self._a1v)
        np.subtract(self['at'][self._idx, :, i], self._a1v, out=self['at'][self._idx, :, i])
        np.multiply(self._relative_r[:, :k], self._a2[None, :k], out=self._a2r[:, :k])
        np.add(
            self['at'][self._idx, :, i + 1 :],
            self._a2r[:, :k],
            out=self['at'][self._idx, :, i + 1 :],
        )

    def step_stage1(self, start: int, stop: int):
        self['at'][self._idx, :, :] = 0.0
        for row in range(start, stop):
            self._update_pair(row, self._length - 1 - row)

    @staticmethod
    def gc_collect():
        gc.collect()


class Universe(UniverseBase):
    __doc__ = __description__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._r = None
        self._a = None
        self._at = None
        self._m = None

        self._segments = None
        self._pool = None

    def start_kernel(self):

        self._segments = Block.get_segments(n = len(self), threads = self._threads)
        self._pool = ShmPool(
            nodes = self._threads,
            worker = UniverseWorker,
            length = len(self),  # custom
            dtype = self._dtype,  # custom
            G = self._G,  # custom
        )

        # Allocate memory: Object parameters
        self._r = self._pool.empty('r', shape = (DIMS, len(self)), dtype=self._dtype.name)
        self._a = np.zeros((DIMS, len(self)), dtype=self._dtype.name)
        self._at = self._pool.empty('at', shape = (self._threads, DIMS, len(self)), dtype=self._dtype.name)
        self._m = self._pool.empty('m', shape = (len(self),), dtype=self._dtype.name)

        # Copy const data into Numpy infrastructure
        for idx, pm in enumerate(self._masses):
            self._m[idx] = pm.m

    def stop_kernel(self):
        self._pool.close()

    def gc_collect(self):
        self._pool.run_all('gc_collect')

    def push_stage1(self):
        for idx, pm in enumerate(self._masses):
            self._r[:, idx] = pm.r[:]

    def step_stage1(self):
        _ = self._pool.run('step_stage1', [Param(segment.start, segment.stop) for segment in self._segments])
        np.add.reduce(self._at, axis = 0, out = self._a)

    def pull_stage1(self):
        for idx, pm in enumerate(self._masses):
            pm.a[:] = [float(v) for v in self._a[:, idx]]
