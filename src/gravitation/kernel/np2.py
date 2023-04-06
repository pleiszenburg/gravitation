# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/np2.py: Kernel

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

__longname__ = 'numpy-backend (2)'
__version__ = '0.0.2'
__description__ = 'numpy backend, optimized array layout, stage 2 also numpy'
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

	def start_kernel(self):
		self.DTYPE = self._dtype
		# Get const values
		self.MASS_LEN = len(self)
		self.SIM_DIM = len(self._mass_list[0]._r)
		# Allocate memory: Object parameters
		self.mass_r_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE, order = 'F')
		self.mass_v_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE, order = 'F')
		self.mass_a_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE, order = 'F')
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
		self.mass_vt_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE, order = 'F')
		self.relative_r = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.DTYPE, order = 'F')
		self.distance_sq = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.distance_sqv = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.DTYPE, order = 'F')
		self.distance_inv = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.a_factor = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.a1 = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.a1r = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.DTYPE, order = 'F')
		self.a1v = np.zeros((self.SIM_DIM,), dtype = self.DTYPE)
		self.a2 = np.zeros((self.MASS_LEN - 1,), dtype = self.DTYPE)
		self.a2r = np.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.DTYPE, order = 'F')

	def update_pair(self, i, k):
		np.subtract(self.mass_r_array[i,:], self.mass_r_array[i+1:,:], out = self.relative_r[:k])
		np.multiply(self.relative_r[:k], self.relative_r[:k], out = self.distance_sqv[:k])
		np.add.reduce(self.distance_sqv[:k], axis = 1, out = self.distance_sq[:k])
		np.sqrt(self.distance_sq[:k], out = self.distance_inv[:k])
		np.divide(1.0, self.distance_inv[:k], out = self.distance_inv[:k])
		np.multiply(self.relative_r[:k], self.distance_inv[:k, None], out = self.relative_r[:k])
		np.divide(self._G, self.distance_sq[:k], out = self.a_factor[:k])
		np.multiply(self.a_factor[:k], self.mass_m_array[i+1:], out = self.a1[:k])
		np.multiply(self.a_factor[:k], self.mass_m_array[i], out = self.a2[:k])
		np.multiply(self.relative_r[:k], self.a1[:k, None], out = self.a1r[:k])
		np.add.reduce(self.a1r[:k], axis = 0, out = self.a1v)
		np.subtract(self.mass_a_array[i,:], self.a1v, out = self.mass_a_array[i,:])
		np.multiply(self.relative_r[:k], self.a2[:k, None], out = self.a2r[:k])
		np.add(self.mass_a_array[i+1:,:], self.a2r[:k], out = self.mass_a_array[i+1:,:])

	def step_stage1(self):
		# Run "pair" calculation: One object against vector of objects per iteration
		for row in range(0, self.MASS_LEN - 1):
			self.update_pair(row, self.MASS_LEN - 1 - row) # max for temp arrays

	def step_stage2(self):
		np.multiply(self.mass_a_array, self._T, out = self.mass_a_array)
		np.add(self.mass_v_array, self.mass_a_array, out = self.mass_v_array)
		np.multiply(self.mass_v_array, self._T, out = self.mass_vt_array)
		np.add(self.mass_r_array, self.mass_vt_array, out = self.mass_r_array)
		self.mass_a_array[:, :] = 0.0
