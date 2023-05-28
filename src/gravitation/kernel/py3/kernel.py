# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/py2.py: Kernel

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

__longname__ = "python-backend (3)"
__version__ = "0.0.1"
__description__ = "pure python backend, significant optimization experiment"
__requirements__ = []
__externalrequirements__ = []
__interpreters__ = ["python3", "pypy3"]
__parallel__ = False
__license__ = "GPLv2"
__authors__ = [
    "Sebastian M. Ernst <ernst@pleiszenburg.de>",
]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from itertools import islice
from math import sqrt

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
        self._m = [pm.m for pm in self._masses]

    def _update_pair(self, i: int):
        # relative_r = self._r[i, :] - self._r[i + 1 :, :]
        ri = [rd[i] for rd in self._r]
        relative_r = [
            [ri - rn for rn in islice(rd, i + 1, None)] for rd, ri in zip(self._r, ri)
        ]

        # relative_r_sq = relative_r ** 2
        relative_r_sq = ((rn**2 for rn in rd) for rd in relative_r)

        # distance_sq = relative_r_sq[:, 0] + relative_r_sq[:, 1] + relative_r_sq[:, 2]
        distance_sq = [sum(rs) for rs in zip(*relative_r_sq)]

        # relative_r = relative_r / np.sqrt(distance_sq)[:, None]
        distance = [sqrt(dn) for dn in distance_sq]
        relative_r = [[rn / dn for rn, dn in zip(rd, distance)] for rd in relative_r]

        # a_factor = self._G / distance_sq
        a_factor = [self._G / dn for dn in distance_sq]

        # a1 = a_factor * self._m[i + 1 :]
        a1 = (an * mn for an, mn in zip(a_factor, islice(self._m, i + 1, None)))

        # a2 = a_factor * self._m[i]
        mi = self._m[i]
        a2 = [an * mi for an in a_factor]

        # self._a[i, :] -= np.sum(relative_r * a1[:, None], axis = 0)
        a = [sum(ri * ad for ri in rs) for rs, ad in zip(zip(*relative_r), a1)]
        for ad, aj in zip(self._a, a):
            ad[i] -= aj

        # self._a[i + 1 :, :] += relative_r * a2[:, None]
        a = [[ri * ai for ri, ai in zip(rd, a2)] for rd in relative_r]
        for ad, aj in zip(self._a, a):
            for idx, ai in enumerate(aj, start=i + 1):
                ad[idx] += ai

    def push_stage1(self):
        self._r = [[pm.r[dim] for pm in self._masses] for dim in range(DIMS)]

    def step_stage1(self):
        self._a = [[0.0 for __ in range(len(self))] for _ in range(DIMS)]

        for row in range(0, len(self) - 1):
            self._update_pair(row)

    def pull_stage1(self):
        for dim, ad in enumerate(self._a):
            for idx, pm in enumerate(self._masses):
                pm.a[dim] = ad[idx]
