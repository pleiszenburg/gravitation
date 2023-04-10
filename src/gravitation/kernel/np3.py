# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/np2.py: Kernel

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
__version__ = "0.0.2"
__description__ = "numpy backend, no memory allocations, optimal memory layout"
__requirements__ = ["numpy"]
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

from ._base import UniverseBase, DIMS

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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

    def start_kernel(self):

        # Allocate memory: Object parameters
        self._r = np.zeros((DIMS, len(self)), dtype=self._dtype)
        self._a = np.zeros((DIMS, len(self)), dtype=self._dtype)
        self._m = np.zeros((len(self),), dtype=self._dtype)

        # Copy const data into Numpy infrastructure
        for idx, pm in enumerate(self._masses):
            self._m[idx] = pm.m

        # Allocate memory: Temporary variables
        self._relative_r = np.zeros((DIMS, len(self) - 1), dtype=self._dtype)
        self._distance_sq = np.zeros((len(self) - 1,), dtype=self._dtype)
        self._distance_sqv = np.zeros(
            (DIMS, len(self) - 1), dtype=self._dtype
        )
        self._distance_inv = np.zeros((len(self) - 1,), dtype=self._dtype)
        self._a_factor = np.zeros((len(self) - 1,), dtype=self._dtype)
        self._a1 = np.zeros((len(self) - 1,), dtype=self._dtype)
        self._a1r = np.zeros((DIMS, len(self) - 1), dtype=self._dtype)
        self._a1v = np.zeros((DIMS,), dtype=self._dtype)
        self._a2 = np.zeros((len(self) - 1,), dtype=self._dtype)
        self._a2r = np.zeros((DIMS, len(self) - 1), dtype=self._dtype)

    def _update_pair(self, i: int, k: int):

        np.subtract(
            self._r[:, i, None],
            self._r[:, i + 1 :],
            out=self._relative_r[:, :k],
        )
        np.multiply(self._relative_r[:, :k], self._relative_r[:, :k], out=self._distance_sqv[:, :k])
        np.add.reduce(self._distance_sqv[:, :k], axis=0, out=self._distance_sq[:k])
        np.sqrt(self._distance_sq[:k], out=self._distance_inv[:k])
        np.divide(1.0, self._distance_inv[:k], out=self._distance_inv[:k])
        np.multiply(
            self._relative_r[:, :k],
            self._distance_inv[None, :k],
            out=self._relative_r[:, :k],
        )
        np.divide(self._G, self._distance_sq[:k], out=self._a_factor[:k])
        np.multiply(self._a_factor[:k], self._m[i + 1 :], out=self._a1[:k])
        np.multiply(self._a_factor[:k], self._m[i], out=self._a2[:k])
        np.multiply(self._relative_r[:, :k], self._a1[None, :k], out=self._a1r[:, :k])
        np.add.reduce(self._a1r[:, :k], axis=1, out=self._a1v)
        np.subtract(self._a[:, i], self._a1v, out=self._a[:, i])
        np.multiply(self._relative_r[:, :k], self._a2[None, :k], out=self._a2r[:, :k])
        np.add(
            self._a[:, i + 1 :],
            self._a2r[:, :k],
            out=self._a[:, i + 1 :],
        )

    def push_stage1(self):

        for idx, pm in enumerate(self._masses):
            self._r[:, idx] = pm.r[:]

    def step_stage1(self):

        self._a[:, :] = 0.0

        for row in range(0, len(self) - 1):
            self._update_pair(row, len(self) - 1 - row)

    def pull_stage1(self):

        for idx, pm in enumerate(self._masses):
            pm.a[:] = [float(v) for v in self._a[:, idx]]
