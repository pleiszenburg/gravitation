# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/np1.py: Kernel

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

__longname__ = "numpy-backend (1)"
__version__ = "0.0.1"
__description__ = "numpy backend, simple"
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

    def start_kernel(self):

        self._r = np.zeros((len(self), DIMS), dtype=self._dtype)
        self._a = np.zeros((len(self), DIMS), dtype=self._dtype)
        self._m = np.zeros((len(self),), dtype=self._dtype)

        for idx, pm in enumerate(self._masses):
            self._m[idx] = pm.m

    def _update_pair(self, i: int):

        relative_r = self._r[i, :] - self._r[i + 1 :, :]
        relative_r_sq = relative_r ** 2
        distance_sq = relative_r_sq[:, 0] + relative_r_sq[:, 1] + relative_r_sq[:, 2]
        relative_r = relative_r / np.sqrt(distance_sq)[:, None]

        a_factor = self._G / distance_sq
        a1 = a_factor * self._m[i + 1 :]
        a2 = a_factor * self._m[i]

        self._a[i, :] -= np.sum(relative_r * a1[:, None], axis = 0)
        self._a[i + 1 :, :] += relative_r * a2[:, None]

    def push_stage1(self):

        for idx, pm in enumerate(self._masses):
            self._r[idx, :] = pm.r[:]

    def step_stage1(self):

        self._a[:, :] = 0.0

        for row in range(0, len(self) - 1):
            self._update_pair(row)

    def pull_stage1(self):

        for idx, pm in enumerate(self._masses):
            pm.a[:] = [float(v) for v in self._a[idx, :]]
