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
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numba as nb
import numpy as np

from gravitation import BaseUniverse
from gravitation import DIMS, Target, Threads
from gravitation import typechecked

from .meta import DESCRIPTION, VARIATIONS

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONFIG
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

_target = VARIATIONS.selected['target']
_threads = VARIATIONS.selected['threads']

if _target is Target.cpu:
    if _threads is Threads.auto or _threads is Threads.single:
        _target = 'cpu'
    else:
        _target = 'parallel'
        if _threads is not Threads.auto:
            nb.set_num_threads(_threads.value)
else:  # gpu:
    _target = 'cuda'

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@nb.guvectorize([
    'void(f4[:,:],f4[:],f4,f4[:,:])',
    'void(f8[:,:],f8[:],f8,f8[:,:])',
], '(m,n),(n),()->(m,n)', nopython = True, target = _target)
def _step_stage1_guv(r, m, g, a):
    for idx in range(0, r.shape[1]):
        for jdx in range(idx + 1, r.shape[1]):
            relative_x = r[0, idx] - r[0, jdx]
            relative_y = r[1, idx] - r[1, jdx]
            relative_z = r[2, idx] - r[2, jdx]
            distance_inv = 1 / np.sqrt(relative_x ** 2 + relative_y ** 2 + relative_z ** 2)
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
class Universe(BaseUniverse):
    __doc__ = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._r = None
        self._a = None
        self._m = None
        self._Gd = None

    def start_kernel(self):
        # Allocate memory: Object parameters
        self._r = np.zeros((DIMS, len(self)), dtype=self._variation.getvalue('dtype'))
        self._a = np.zeros((DIMS, len(self)), dtype=self._variation.getvalue('dtype'))
        self._m = np.zeros((len(self),), dtype=self._variation.getvalue('dtype'))

        # Copy const data into Numpy infrastructure
        for idx, pm in enumerate(self._masses):
            self._m[idx] = pm.m

        self._Gd = getattr(np, self._variation.getvalue('dtype'))(self._G)

    def push_stage1(self):
        for idx, pm in enumerate(self._masses):
            self._r[:, idx] = pm.r[:]

    def step_stage1(self):
        self._a[:, :] = 0.0

        _step_stage1_guv(self._r, self._m, self._Gd, self._a)

    def pull_stage1(self):
        for idx, pm in enumerate(self._masses):
            pm.a[:] = [float(v) for v in self._a[:, idx]]
