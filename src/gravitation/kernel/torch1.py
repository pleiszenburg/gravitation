# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/torch1.py: Kernel

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

__longname__ = 'pytorch-backend'
__version__ = '0.0.1'
__description__ = 'using pytorch as a numpy-like cpu/gpu-agnostic backend'
__requirements__ = ['numpy', 'torch']
__externalrequirements__ = ['cuda']
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
import torch

from ._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class universe(universe_base):

	def start_kernel(self):
		self.DTYPE = self._dtype
		self.TDTYPE = getattr(torch, self._dtype)
		# Get const values
		self.MASS_LEN = len(self)
		self.SIM_DIM = len(self._mass_list[0]._r)
		# Allocate memory: Object parameters
		self.mass_r_arrayc = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE)
		self.mass_r_arrayg = torch.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.TDTYPE)
		self.mass_a_arrayc = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE)
		self.mass_a_arrayg = torch.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.TDTYPE)
		self.mass_m_array = torch.zeros((self.MASS_LEN,), dtype = self.TDTYPE)
		# Copy const data into Numpy infrastructure
		for pm_index, pm in enumerate(self._mass_list):
			self.mass_m_array[pm_index] = pm._m
		# Allocate memory: Temporary variables
		self.relative_r = torch.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.TDTYPE)
		self.distance_sq = torch.zeros((self.MASS_LEN - 1,), dtype = self.TDTYPE)
		self.distance_sqv = torch.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.TDTYPE)
		self.distance_inv = torch.zeros((self.MASS_LEN - 1,), dtype = self.TDTYPE)
		self.a_factor = torch.zeros((self.MASS_LEN - 1,), dtype = self.TDTYPE)
		self.a1 = torch.zeros((self.MASS_LEN - 1,), dtype = self.TDTYPE)
		self.a1r = torch.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.TDTYPE)
		self.a1v = torch.zeros((self.SIM_DIM,), dtype = self.TDTYPE)
		self.a2 = torch.zeros((self.MASS_LEN - 1,), dtype = self.TDTYPE)
		self.a2r = torch.zeros((self.MASS_LEN - 1, self.SIM_DIM), dtype = self.TDTYPE)

	def update_pair(self, i, k):
		torch.sub(self.mass_r_arrayg[i,:], self.mass_r_arrayg[i+1:,:], out = self.relative_r[:k])
		torch.mul(self.relative_r[:k], self.relative_r[:k], out = self.distance_sqv[:k])
		# np.add.reduce(self.distance_sqv[:k], axis = 1, out = self.distance_sq[:k])
		torch.sum(self.distance_sqv[:k], dim = 1, out = self.distance_sq[:k])
		torch.sqrt(self.distance_sq[:k], out = self.distance_inv[:k])

		# torch.div(1.0, self.distance_inv[:k], out = self.distance_inv[:k])
		self.distance_inv[:k].pow_(-1)

		torch.mul(self.relative_r[:k], self.distance_inv[:k].reshape(k, 1), out = self.relative_r[:k])

		# torch.div(self._G, self.distance_sq[:k], out = self.a_factor[:k])
		torch.mul(self.distance_sq[:k], 1. / self._G, out = self.a_factor[:k])
		self.a_factor[:k].pow_(-1)

		torch.mul(self.a_factor[:k], self.mass_m_array[i+1:], out = self.a1[:k])
		torch.mul(self.a_factor[:k], self.mass_m_array[i], out = self.a2[:k])
		torch.mul(self.relative_r[:k], self.a1[:k].reshape(k, 1), out = self.a1r[:k])
		# np.add.reduce(self.a1r[:k], axis = 0, out = self.a1v)
		torch.sum(self.a1r[:k], dim = 0, out = self.a1v)
		torch.sub(self.mass_a_arrayg[i,:], self.a1v, out = self.mass_a_arrayg[i,:])
		torch.mul(self.relative_r[:k], self.a2[:k].reshape(k, 1), out = self.a2r[:k])

		# torch.add(self.mass_a_arrayg[i+1:,:], self.a2r[:k,:], self.mass_a_arrayg[i+1:,:])
		self.mass_a_arrayg[i+1:,:].add_(self.a2r[:k,:])

	def step_stage1(self):
		# Zero out variables
		self.mass_a_arrayg[:, :] = 0.0
		# Copy dynamic data to Numpy infrastructure
		for pm_index, pm in enumerate(self._mass_list):
			self.mass_r_arrayc[pm_index,:] = pm._r[:]
		# Push data to graphics card
		self.mass_r_arrayg[:,:] = torch.from_numpy(self.mass_r_arrayc)
		# Run "pair" calculation: One object against vector of objects per iteration
		for row in range(0, self.MASS_LEN - 1):
			self.update_pair(row, self.MASS_LEN - 1 - row) # max for temp arrays
		# Fetch data from graphics card
		self.mass_a_arrayc[:,:] = self.mass_a_arrayg.numpy()
		# Push dynamic data back to Python infrastructure
		for pm_index, pm in enumerate(self._mass_list):
			pm._a[:] = self.mass_a_arrayc[pm_index,:]
