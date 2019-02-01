# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/nb1.py: Kernel

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

__longname__ = 'numba-numpy-backend (1)'
__version__ = '0.0.1'
__description__ = 'numba-accelerated numpy-implementation, stage 2 numpy'
__requirements__ = ['numba', 'numpy']
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

import numba

import numpy as np

from ._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class universe(universe_base):

	def start_kernel(self):
		self.DTYPE = self._dtype
		# Get const values
		self.MASS_LEN = len(self)
		self.SIM_DIM = len(self._mass_list[0]._r)
		# Allocate memory: Object parameters
		self.mass_r_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE)
		self.mass_v_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE)
		self.mass_a_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE)
		self.mass_m_array = np.zeros((self.MASS_LEN,), dtype = self.DTYPE)
		# Copy const data into Numpy infrastructure and link mass objects to Numpy views
		for pm_index, pm in enumerate(self._mass_list):
			self.mass_m_array[pm_index] = pm._m
		for pm_index, pm in enumerate(self._mass_list):
			self.mass_r_array[pm_index,:] = pm._r[:]
			pm._r = self.mass_r_array[pm_index,:]
			self.mass_v_array[pm_index,:] = pm._v[:]
			pm._v = self.mass_v_array[pm_index,:]
			pm._a = self.mass_a_array[pm_index,:]
		# Allocate memory: Temporary variables
		self.mass_vt_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE)
		self.relative_r = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.DTYPE)
		self.distance_sq = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.distance_sqv = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.DTYPE)
		self.distance_inv = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.a_factor = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.a1 = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.a1r = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.DTYPE)
		self.a1v = np.zeros((self.SIM_DIM,), dtype = self.DTYPE)
		self.a2 = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.a2r = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.DTYPE)

	@staticmethod
	@numba.jit(nopython = True)
	def step_stage1_jit(
		mass_r_array, mass_a_array, mass_m_array,
		relative_r, distance_sqv, distance_sq,
		distance_inv, a_factor, a1, a2, a1r, a1v, a2r,
		MASS_LEN, _G,
		):
		mass_a_array[:,:] = 0.0
		for row in range(0, MASS_LEN - 1):
			i, k = row, MASS_LEN - 1 - row
			np.subtract(mass_r_array[i,:], mass_r_array[i+1:,:], relative_r[:k])
			np.multiply(relative_r[:k], relative_r[:k], distance_sqv[:k])

			# np.add.reduce(distance_sqv[:k], axis = 1, out = distance_sq[:k])
			# np.sum(distance_sqv[:k], 1, distance_sqv.dtype, distance_sq[:k])
			distance_sq[:k] = np.sum(distance_sqv[:k], 1)

			np.sqrt(distance_sq[:k], distance_inv[:k])
			np.divide(1.0, distance_inv[:k], distance_inv[:k])
			np.multiply(relative_r[:k], distance_inv[:k].reshape(k, 1), relative_r[:k])
			np.divide(_G, distance_sq[:k], a_factor[:k])
			np.multiply(a_factor[:k], mass_m_array[i+1:], a1[:k])
			np.multiply(a_factor[:k], mass_m_array[i], a2[:k])
			np.multiply(relative_r[:k], a1[:k].reshape(k, 1), a1r[:k])

			# np.add.reduce(a1r[:k], axis = 0, out = a1v)
			# np.sum(a1r[:k], 0, a1r.dtype, a1v)
			a1v[:] = np.sum(a1r[:k], 0)

			np.subtract(mass_a_array[i,:], a1v, mass_a_array[i,:])
			np.multiply(relative_r[:k], a2[:k].reshape(k, 1), a2r[:k])
			np.add(mass_a_array[i+1:,:], a2r[:k], mass_a_array[i+1:,:])

	def step_stage1(self):
		universe.step_stage1_jit(
			self.mass_r_array, self.mass_a_array, self.mass_m_array,
			self.relative_r, self.distance_sqv, self.distance_sq,
			self.distance_inv, self.a_factor, self.a1, self.a2, self.a1r, self.a1v, self.a2r,
			self.MASS_LEN, self._G,
			)

	def step_stage2(self):
		np.multiply(self.mass_a_array, self._T, out = self.mass_a_array)
		np.add(self.mass_v_array, self.mass_a_array, out = self.mass_v_array)
		np.multiply(self.mass_v_array, self._T, out = self.mass_vt_array)
		np.add(self.mass_r_array, self.mass_vt_array, out = self.mass_r_array)
		self.mass_a_array[:, :] = 0.0
