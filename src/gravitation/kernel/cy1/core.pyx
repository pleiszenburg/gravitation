# -*- coding: utf-8 -*-
# cython: language_level=3

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/cy1/core.pyx: cy1 kernel cython core

	Copyright (C) 2019 Sebastian M. Ernst <ernst@pleiszenburg.de>

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

from math import sqrt

from .._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class universe(universe_base):

	def update_pair(self, pm1, pm2):
		relative_r = [(r1 - r2) for r1, r2 in zip(pm1._r, pm2._r)]
		distance_sq = sum([r ** 2 for r in relative_r])
		distance_inv = 1.0 / sqrt(distance_sq)
		relative_r = [r * distance_inv for r in relative_r]
		a_factor = self._G / distance_sq
		a1 = a_factor * pm2._m
		a2 = a_factor * pm1._m
		pm1._a[:] = [a - r * a1 for r, a in zip(relative_r, pm1._a)]
		pm2._a[:] = [a + r * a2 for r, a in zip(relative_r, pm2._a)]

	def step_stage1(self):
		for pm1_index, pm1 in enumerate(self._mass_list[:-1]):
			for pm2_index, pm2 in enumerate(self._mass_list[pm1_index+1:]):
				self.update_pair(pm1, pm2)
