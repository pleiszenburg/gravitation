# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/np2/kernel.py: Kernel

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

__longname__ = "numba-numpy-backend (3)"
__version__ = "0.0.1"
__description__ = "numba guvectorize-kernel"
__requirements__ = ["numba", "numpy"]
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

from math import sqrt

import numba as nb
import numpy as np

from ...lib.base import UniverseBase
from ...lib.const import DIMS
from ...lib.debug import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@nb.guvectorize([
    'void(f4[:,:],f4[:],f4,f4[:,:])',
    'void(f8[:,:],f8[:],f8,f8[:,:])',
], '(m,n),(n),()->(m,n)', nopython = True, target = 'cpu')
def _step_stage1_guv(r, m, g, a):
    for idx in range(0, r.shape[1]):
        for jdx in range(idx + 1, r.shape[1]):
            relative_x = r[0, idx] - r[0, jdx]
            relative_y = r[1, idx] - r[1, jdx]
            relative_z = r[2, idx] - r[2, jdx]
            distance_inv = 1 / sqrt(relative_x ** 2 + relative_y ** 2 + relative_z ** 2)
            relative_x *= distance_inv
            relative_y *= distance_inv
            relative_z *= distance_inv
            distance_inv **= 2
            distance_inv *= g
            a1 = m[jdx] * distance_inv
            a2 = m[idx] * distance_inv
            a[0, idx] -= relative_x * a1
            a[1, idx] -= relative_y * a1
            a[2, idx] -= relative_z * a1
            a[0, jdx] += relative_x * a2
            a[1, jdx] += relative_y * a2
            a[2, jdx] += relative_z * a2

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

    def start_kernel(self):
        # Allocate memory: Object parameters
        self._r = np.zeros((DIMS, len(self)), dtype=self._dtype.name)
        self._a = np.zeros((DIMS, len(self)), dtype=self._dtype.name)
        self._m = np.zeros((len(self),), dtype=self._dtype.name)

        # Copy const data into Numpy infrastructure
        for idx, pm in enumerate(self._masses):
            self._m[idx] = pm.m

    def push_stage1(self):
        for idx, pm in enumerate(self._masses):
            self._r[:, idx] = pm.r[:]

    def step_stage1(self):
        self._a[:, :] = 0.0

        _step_stage1_guv(self._r, self._m, self._G, self._a)

    def pull_stage1(self):
        for idx, pm in enumerate(self._masses):
            pm.a[:] = [float(v) for v in self._a[:, idx]]
