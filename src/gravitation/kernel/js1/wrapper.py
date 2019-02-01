# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/js1/wrapper.py: js1 kernel core wrapper

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

import json
from math import sqrt
import os

from py_mini_racer import py_mini_racer

from .._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class universe(universe_base):

	def start_kernel(self):
		self.ctx = py_mini_racer.MiniRacer()
		with open(os.path.join(os.path.dirname(__file__), 'core.js'), 'r') as f:
			self.ctx.eval(
				f.read().replace(
					'__SIM_DIM__', '%d' % len(self._mass_list[0]._r)
					).replace(
					'__G__', '%e' % self._G
					)
				)
		self.ctx.eval('var w = new universe({mass_list:s}, {G:e});'.format(
			mass_list = json.dumps([pm._m for pm in self._mass_list]),
			G = self._G,
			))

	def step_stage1(self):
		a = self.ctx.eval('w.step_stage1({r:s})'.format(
			r = json.dumps([pm._r for pm in self._mass_list])
			))
		for pm, a_item in zip(self._mass_list, a):
			pm._a[:] = a_item[:]
