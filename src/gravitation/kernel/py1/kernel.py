# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/py1/kernel.py: Kernel

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

import math

from . import DESCRIPTION

from gravitation import BaseUniverse
from gravitation import Dtype
from gravitation import typechecked
from gravitation import VariationError
from gravitation import PointMass

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Universe(BaseUniverse):
    __doc__ = DESCRIPTION

    def start_kernel(self):
        if self._variation['dtype'] is Dtype.float32:
            raise VariationError('kernel does not support float32')

    def _update_pair(self, pm1: PointMass, pm2: PointMass):
        relative_r = [(r1 - r2) for r1, r2 in zip(pm1.r, pm2.r)]
        distance = math.sqrt(sum([r**2 for r in relative_r]))
        relative_r = [r / distance for r in relative_r]
        a1 = self._G * pm2.m / (distance**2)
        a2 = self._G * pm1.m / (distance**2)
        pm1.a[:] = [a - r * a1 for r, a in zip(relative_r, pm1.a)]
        pm2.a[:] = [a + r * a2 for r, a in zip(relative_r, pm2.a)]

    def step_stage1(self):
        for pm1_index, pm1 in enumerate(self._masses[:-1]):
            for pm2 in self._masses[pm1_index + 1 :]:
                self._update_pair(pm1, pm2)
