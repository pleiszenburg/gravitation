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

__longname__ = "python-backend (2)"
__version__ = "0.1.0"
__description__ = "pure python backend, small optimization experiment"
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

import math

from ._base import UniverseBase, PointMass

from typeguard import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Universe(UniverseBase):
    __doc__ = __description__

    def _update_pair(self, pm1: PointMass, pm2: PointMass):
        relative_r = [(r1 - r2) for r1, r2 in zip(pm1.r, pm2.r)]
        distance_sq = sum([r**2 for r in relative_r])
        distance_inv = 1.0 / math.sqrt(distance_sq)
        relative_r = [r * distance_inv for r in relative_r]
        a_factor = self._G / distance_sq
        a1 = a_factor * pm2.m
        a2 = a_factor * pm1.m
        pm1.a[:] = [a - r * a1 for r, a in zip(relative_r, pm1.a)]
        pm2.a[:] = [a + r * a2 for r, a in zip(relative_r, pm2.a)]

    def step_stage1(self):
        for pm1_index, pm1 in enumerate(self._masses[:-1]):
            for pm2 in self._masses[pm1_index + 1 :]:
                self._update_pair(pm1, pm2)
