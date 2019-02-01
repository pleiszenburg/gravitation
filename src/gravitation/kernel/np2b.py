# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/np2b.py: Kernel

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

__longname__ = 'numpy-backend (2b)'
__version__ = '0.0.1'
__description__ = 'numpy backend, optimized array layout, stage 2 also numpy, CPU-cache experiment'
__requirements__ = ['numpy']
__externalrequirements__ = []
__interpreters__ = ['python3']
__parallel__ = False
__license__ = 'GPLv2'
__authors__ = [
	'Sebastian M. Ernst <ernst@pleiszenburg.de>',
	]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

from ._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class universe(universe_base):

	def _calloc(self, *size):
		if len(size) > 1:
			return np.zeros(size, dtype = self.DTYPE, order = 'F')
		else:
			return np.zeros(size, dtype = self.DTYPE)

	@staticmethod
	def _chunks(start, stop, intv):
		a = list(zip(range(start, stop, intv), range(start + intv, stop + intv, intv)))
		if a[-1][1] != stop:
			a[-1] = (a[-1][0], stop)
		return a

	def start_kernel(self):
		self.DTYPE = self._dtype
		# Get const values
		self.MASS_LEN = len(self)
		self.SIM_DIM = len(self._mass_list[0]._r)
		# Segmentation (CPU cache optimization)
		self.COLS = 1024
		self.COLS_LIST = universe._chunks(0, self.MASS_LEN, self.COLS)
		# Allocate memory: Object parameters
		self.mass_r_array = self._calloc(self.MASS_LEN, self.SIM_DIM)
		self.mass_v_array = self._calloc(self.MASS_LEN, self.SIM_DIM)
		self.mass_a_array = self._calloc(self.MASS_LEN, self.SIM_DIM)
		self.mass_m_array = self._calloc(self.MASS_LEN)
		# Copy const data into Numpy infrastructure and link mass objects to Numpy views
		for pm_index, pm in enumerate(self._mass_list):
			self.mass_m_array[pm_index] = pm._m
		for pm_index, pm in enumerate(self._mass_list):
			self.mass_r_array[pm_index,:] = pm._r[:]
			pm._r = self.mass_r_array[pm_index,:]
			self.mass_v_array[pm_index,:] = pm._v[:]
			pm._v = self.mass_v_array[pm_index,:]
			pm._a = self.mass_a_array[pm_index,:]
		# Allocate memory: Temporary variables - Stage 1
		self.relative_r = self._calloc(self.COLS, self.SIM_DIM)
		self.distance_sq = self._calloc(self.COLS)
		self.distance_sqv = self._calloc(self.COLS, self.SIM_DIM)
		self.distance_inv = self._calloc(self.COLS)
		self.a_factor = self._calloc(self.COLS)
		self.a1 = self._calloc(self.COLS)
		self.a1r = self._calloc(self.COLS, self.SIM_DIM)
		self.a1v = self._calloc(self.SIM_DIM)
		self.a2 = self._calloc(self.COLS)
		self.a2r = self._calloc(self.COLS, self.SIM_DIM)
		# Allocate memory: Temporary variables - Stage 2
		self.mass_vt_array = self._calloc(self.MASS_LEN, self.SIM_DIM)

	def update_pair(self, item, b_a, b_z):
		b_len = b_z - b_a
		np.subtract(self.mass_r_array[item,:], self.mass_r_array[b_a:b_z,:], out = self.relative_r[:b_len])
		np.multiply(self.relative_r[:b_len], self.relative_r[:b_len], out = self.distance_sqv[:b_len])
		np.add.reduce(self.distance_sqv[:b_len], axis = 1, out = self.distance_sq[:b_len])
		np.sqrt(self.distance_sq[:b_len], out = self.distance_inv[:b_len])
		np.divide(1.0, self.distance_inv[:b_len], out = self.distance_inv[:b_len])
		np.multiply(self.relative_r[:b_len], self.distance_inv[:b_len].reshape(b_len, 1), out = self.relative_r[:b_len])
		np.divide(self._G, self.distance_sq[:b_len], out = self.a_factor[:b_len])
		np.multiply(self.a_factor[:b_len], self.mass_m_array[b_a:b_z], out = self.a1[:b_len])
		np.multiply(self.a_factor[:b_len], self.mass_m_array[item], out = self.a2[:b_len])
		np.multiply(self.relative_r[:b_len], self.a1[:b_len].reshape(b_len, 1), out = self.a1r[:b_len])
		np.add.reduce(self.a1r[:b_len], axis = 0, out = self.a1v)
		np.subtract(self.mass_a_array[item,:], self.a1v, out = self.mass_a_array[item,:])
		np.multiply(self.relative_r[:b_len], self.a2[:b_len].reshape(b_len, 1), out = self.a2r[:b_len])
		np.add(self.mass_a_array[b_a:b_z,:], self.a2r[:b_len], out = self.mass_a_array[b_a:b_z,:])

	def step_stage1(self):
		# Run "pair" calculation: One object against vector of objects per iteration
		for col_min, col_max in self.COLS_LIST:
			for row in range(0, col_max - 1):
				self.update_pair(
					item = row,
					b_a = col_min if row < col_min else row + 1,
					b_z = col_max,
					)

	def step_stage2(self):
		np.multiply(self.mass_a_array, self._T, out = self.mass_a_array)
		np.add(self.mass_v_array, self.mass_a_array, out = self.mass_v_array)
		np.multiply(self.mass_v_array, self._T, out = self.mass_vt_array)
		np.add(self.mass_r_array, self.mass_vt_array, out = self.mass_r_array)
		self.mass_a_array[:, :] = 0.0
