# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/oc4/wrapper.py: oc4 kernel core wrapper

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
import multiprocessing as mp
import os

import numpy as np

import oct2py

from .._base_ import universe_base

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def create_worker_context():
	# Store context information in global namespace of worker
	global context
	context = {}

def init_worker_context(worker_id, m, G, SIM_DIM, MASS_LEN):
	# Set up "shared memory path" for octave
	oc_path = '/dev/shm/oc%02d' % worker_id
	try:
		os.mkdir(oc_path)
	except FileExistsError:
		pass
	# Start octave sessions
	oc = oct2py.Oct2Py(
		temp_dir = oc_path,
		convert_to_float = False,
		)
	oc.addpath(os.path.dirname(__file__))
	for name, var in [
		('m', m),
		('G', G),
		('SIM_DIM', SIM_DIM),
		('MASS_LEN', MASS_LEN),
		]:
		oc.eval('global %s' % name)
		oc.push(name, var, verbose = True)
	# Store context information in global namespace of worker
	context.update({
		'oc': oc,
		})
	return True

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class universe(universe_base):

	def start_kernel(self):

		self.DTYPE = self._dtype

		# Get const values
		self.MASS_LEN = len(self)
		self.SIM_DIM = len(self._mass_list[0]._r)
		self.CPU_LEN = self._threads
		# Allocate memory: Object parameters
		self.mass_r_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype = self.DTYPE)
		mass_m_array = np.zeros((self.MASS_LEN,), dtype = self.DTYPE)
		# Copy const data into Numpy infrastructure
		for pm_index, pm in enumerate(self._mass_list):
			mass_m_array[pm_index] = pm._m

		# Compute line index tuples for evenly sized batches
		total_pairs = (self.MASS_LEN * (self.MASS_LEN - 1)) // 2
		batch_length = total_pairs // self.CPU_LEN
		self.index_pool = []
		pair_count = 0
		start_line = 0
		for line in range(1, self.MASS_LEN - 1):
			pair_count += (self.MASS_LEN - 1 - line)
			if pair_count < batch_length:
				continue
			pair_count = 0
			self.index_pool.append((start_line, line))
			start_line = line
		assert len(self.index_pool) in [(self.CPU_LEN - 1), self.CPU_LEN]
		if len(self.index_pool) == self.CPU_LEN - 1:
			self.index_pool.append((start_line, self.MASS_LEN - 1))
		assert self.index_pool[-1][1] == self.MASS_LEN - 1

		# Init multiprocessing pool
		self.cpu_pool = mp.Pool(processes = self.CPU_LEN, initializer = create_worker_context)
		pool_results = [
			self.cpu_pool.apply_async(
				init_worker_context,
				args = (core_id, mass_m_array, self._G, self.SIM_DIM, self.MASS_LEN)
			) for core_id in range(self.CPU_LEN)
			]
		result_batches = [result.get() for result in pool_results]
		if not all(result_batches):
			raise SyntaxError('Workers could not be initialized ...')

	@staticmethod
	def step_stage1_batch(i_start, i_end, mass_r_array):
		# Call octave core
		return context['oc'].step_stage1(i_start + 1, i_end, mass_r_array)

	def step_stage1(self):
		# Copy dynamic data to Numpy infrastructure
		for pm_index, pm in enumerate(self._mass_list):
			self.mass_r_array[pm_index,:] = pm._r[:]
		# Run "pair" calculation: One object against vector of objects per iteration
		pool_results = [
			self.cpu_pool.apply_async(
				universe.step_stage1_batch,
				args = (i_start, i_end, self.mass_r_array,)
			) for i_start, i_end in self.index_pool
			]
		result_batches = [result.get() for result in pool_results]
		# Reduce batches
		for batch in result_batches[1:]:
			np.add(result_batches[0], batch, out = result_batches[0])
		# Push dynamic data back to Python infrastructure
		for pm_index, pm in enumerate(self._mass_list):
			pm._a[:] = result_batches[0][pm_index,:]
