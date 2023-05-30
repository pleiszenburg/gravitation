# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/np_p1/kernel.py: Kernel

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

__longname__ = "numpy-backend (parallel 1)"
__version__ = "0.0.2"
__description__ = "numpy backend, no memory allocations, optimal memory layout, thread-based"
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

import sys
from threading import Thread

import numpy as np

from ...lib.base import UniverseBase
from ...lib.block import Block
from ...lib.const import DIMS
from ...lib.debug import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Universe(UniverseBase):
    __doc__ = __description__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._r = None
        self._a = None
        self._m = None

        self._relative_r = None
        self._distance_sq = None
        self._distance_sqv = None
        self._distance_inv = None
        self._a_factor = None
        self._a1 = None
        self._a1r = None
        self._a1v = None
        self._a2 = None
        self._a2r = None
        self._at = None

        self._segments = None

    def start_kernel(self):
        # Allocate memory: Object parameters
        self._r = np.zeros((DIMS, len(self)), dtype=self._dtype.name)
        self._a = np.zeros((DIMS, len(self)), dtype=self._dtype.name)
        self._m = np.zeros((len(self),), dtype=self._dtype.name)

        # Copy const data into Numpy infrastructure
        for idx, pm in enumerate(self._masses):
            self._m[idx] = pm.m

        # Allocate memory: Temporary variables
        self._relative_r = np.zeros((DIMS, len(self) - 1, self._threads), dtype=self._dtype.name)
        self._distance_sq = np.zeros((len(self) - 1, self._threads), dtype=self._dtype.name)
        self._distance_sqv = np.zeros((DIMS, len(self) - 1, self._threads), dtype=self._dtype.name)
        self._distance_inv = np.zeros((len(self) - 1, self._threads), dtype=self._dtype.name)
        self._a_factor = np.zeros((len(self) - 1, self._threads), dtype=self._dtype.name)
        self._a1 = np.zeros((len(self) - 1, self._threads), dtype=self._dtype.name)
        self._a1r = np.zeros((DIMS, len(self) - 1, self._threads), dtype=self._dtype.name)
        self._a1v = np.zeros((DIMS, self._threads), dtype=self._dtype.name)
        self._a2 = np.zeros((len(self) - 1, self._threads), dtype=self._dtype.name)
        self._a2r = np.zeros((DIMS, len(self) - 1, self._threads), dtype=self._dtype.name)
        self._at = np.zeros((DIMS, len(self), self._threads), dtype=self._dtype.name)

        self._segments = Block.get_segments(n = len(self), threads = self._threads)

        # https://tenthousandmeters.com/blog/python-behind-the-scenes-13-the-gil-and-its-effects-on-python-multithreading/
        sys.setswitchinterval(1e-3)

    def _update_pair(self, i: int, k: int, thread: int):
        np.subtract(
            self._r[:, i, None],
            self._r[:, i + 1 :],
            out=self._relative_r[:, :k, thread],
        )
        np.multiply(
            self._relative_r[:, :k, thread],
            self._relative_r[:, :k, thread],
            out=self._distance_sqv[:, :k, thread],
        )
        np.add.reduce(self._distance_sqv[:, :k, thread], axis=0, out=self._distance_sq[:k, thread])
        np.sqrt(self._distance_sq[:k, thread], out=self._distance_inv[:k, thread])
        np.divide(1.0, self._distance_inv[:k, thread], out=self._distance_inv[:k, thread])
        np.multiply(
            self._relative_r[:, :k, thread],
            self._distance_inv[None, :k, thread],
            out=self._relative_r[:, :k, thread],
        )
        np.divide(self._G, self._distance_sq[:k, thread], out=self._a_factor[:k, thread])
        np.multiply(self._a_factor[:k, thread], self._m[i + 1 :], out=self._a1[:k, thread])
        np.multiply(self._a_factor[:k, thread], self._m[i], out=self._a2[:k, thread])
        np.multiply(self._relative_r[:, :k, thread], self._a1[None, :k, thread], out=self._a1r[:, :k, thread])
        np.add.reduce(self._a1r[:, :k, thread], axis=1, out=self._a1v[:, thread])
        np.subtract(self._at[:, i, thread], self._a1v[:, thread], out=self._at[:, i, thread])
        np.multiply(self._relative_r[:, :k, thread], self._a2[None, :k, thread], out=self._a2r[:, :k, thread])
        np.add(
            self._at[:, i + 1 :, thread],
            self._a2r[:, :k, thread],
            out=self._at[:, i + 1 :, thread],
        )

    def push_stage1(self):
        for idx, pm in enumerate(self._masses):
            self._r[:, idx] = pm.r[:]

    def _step_stage1_thread(self, start: int, stop: int, thread: int):
        for row in range(start, stop):
            self._update_pair(row, len(self) - 1 - row, thread)

    def step_stage1(self):
        self._at[:, :, :] = 0.0

        threads = []
        for idx, segment in enumerate(self._segments):
            thread = Thread(
                target = self._step_stage1_thread,
                args = (segment.start, segment.stop, idx),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        np.add.reduce(self._at, axis = 2, out = self._a)

    def pull_stage1(self):
        for idx, pm in enumerate(self._masses):
            pm.a[:] = [float(v) for v in self._a[:, idx]]
