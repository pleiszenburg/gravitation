# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/py1.py: Kernel

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
# KERNEL META
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__longname__ = 'python-backend (1)'
__version__ = '0.0.1'
__description__ = 'pure python backend, reference kernel'
__requirements__ = []
__externalrequirements__ = []
__interpreters__ = ['python3', 'pypy3']
__parallel__ = False
__license__ = 'GPLv2'
__authors__ = [
	'Sebastian M. Ernst <ernst@pleiszenburg.de>',
	]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math

from ._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class universe(universe_base):

	def update_pair(self, pm1, pm2):
		relative_r = [(r1 - r2) for r1, r2 in zip(pm1._r, pm2._r)]
		distance = math.sqrt(sum([r ** 2 for r in relative_r]))
		relative_r = [r / distance for r in relative_r]
		a1 = self._G * pm2._m / (distance ** 2)
		a2 = self._G * pm1._m / (distance ** 2)
		pm1._a[:] = [a - r * a1 for r, a in zip(relative_r, pm1._a)]
		pm2._a[:] = [a + r * a2 for r, a in zip(relative_r, pm2._a)]

	def step_stage1(self):
		for pm1_index, pm1 in enumerate(self._mass_list[:-1]):
			for pm2_index, pm2 in enumerate(self._mass_list[pm1_index+1:]):
				self.update_pair(pm1, pm2)
