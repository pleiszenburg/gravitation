# -*- coding: utf-8 -*-
# cython: language_level=3, boundscheck=False

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/cy4/core.pyx: cy4 kernel cython core

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

import cython

from cython.parallel import prange

from cpython cimport array
import array

from libc.string cimport memset
from libc.math cimport sqrt

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@cython.cdivision(True)
cdef inline void _update_pair_c_(
	long index_i, long index_j,
	long index_i_off, long index_j_off,
	float *rx, float *ry, float *rz,
	float *ax, float *ay, float *az,
	float *m,
	long SIM_DIM,
	float G,
	) nogil:

	cdef float relative_rx, relative_ry, relative_rz
	cdef float distance_sq, distance_inv, a_factor, a1, a2
	cdef float one = 1.0

	relative_rx = rx[index_i] - rx[index_j]
	relative_ry = ry[index_i] - ry[index_j]
	relative_rz = rz[index_i] - rz[index_j]

	distance_sq = (
		(relative_rx * relative_rx)
		+ (relative_ry * relative_ry)
		+ (relative_rz * relative_rz)
		)

	distance_inv = one / sqrt(distance_sq)

	relative_rx *= distance_inv
	relative_ry *= distance_inv
	relative_rz *= distance_inv

	a_factor = G / distance_sq

	a1 = a_factor * m[index_j]
	a2 = a_factor * m[index_i]

	ax[index_i_off] -= relative_rx * a1
	ay[index_i_off] -= relative_ry * a1
	az[index_i_off] -= relative_rz * a1

	ax[index_j_off] += relative_rx * a2
	ay[index_j_off] += relative_ry * a2
	az[index_j_off] += relative_rz * a2

@cython.cdivision(True)
cdef inline void _thread_worker_c_(
	long worker_id,
	float *rx, float *ry, float *rz,
	float *ax, float *ay, float *az,
	float *m,
	long *index_0, long *index_1,
	long SIM_DIM,
	float G,
	) nogil:

	# iteration index variables
	cdef long index_i, index_j

	index_i = index_0[worker_id]
	while index_i < index_1[worker_id]:
		index_j = index_i + 1
		while index_j < SIM_DIM:
			_update_pair_c_(
				index_i, index_j,
				index_i + SIM_DIM * worker_id, index_j + SIM_DIM * worker_id,
				rx, ry, rz,
				ax, ay, az,
				m,
				SIM_DIM,
				G,
				)
			index_j += 1
		index_i += 1

cdef void _step_stage1_c_(
	float *rx, float *ry, float *rz,
	float *ax, float *ay, float *az,
	float *axmp, float *aymp, float *azmp,
	float *m,
	long *index_0, long *index_1,
	long CPU_LEN,
	long SIM_DIM,
	float G,
	):

	# iteration index variables
	cdef long index, index_off, offset, sim_len, mp_sim_len
	cdef long worker_id, off_id

	# reset a to zero
	sim_len = SIM_DIM * sizeof(float)
	memset(ax, 0, sim_len)
	memset(ay, 0, sim_len)
	memset(az, 0, sim_len)
	mp_sim_len = CPU_LEN * sim_len
	memset(axmp, 0, mp_sim_len)
	memset(aymp, 0, mp_sim_len)
	memset(azmp, 0, mp_sim_len)

	# update all unique pairs
	for worker_id in prange(
		0, CPU_LEN,
		schedule = 'static',
		chunksize = 1,
		num_threads = CPU_LEN,
		nogil = True,
		):

		_thread_worker_c_(
			worker_id,
			rx, ry, rz,
			axmp, aymp, azmp,
			m,
			index_0, index_1,
			SIM_DIM,
			G,
			)

	for off_id in range(0, CPU_LEN):
		offset = SIM_DIM * off_id
		for index in range(0, SIM_DIM):
			index_off = offset + index
			ax[index] += axmp[index_off]
			ay[index] += aymp[index_off]
			az[index] += azmp[index_off]

def _step_stage1_(
	array.array rx, array.array ry, array.array rz,
	array.array ax, array.array ay, array.array az,
	array.array axmp, array.array aymp, array.array azmp,
	array.array m,
	array.array index_0, array.array index_1,
	long CPU_LEN,
	long SIM_DIM,
	float G,
	):

	_step_stage1_c_(
		rx.data.as_floats, ry.data.as_floats, rz.data.as_floats,
		ax.data.as_floats, ay.data.as_floats, az.data.as_floats,
		axmp.data.as_floats, aymp.data.as_floats, azmp.data.as_floats,
		m.data.as_floats,
		index_0.data.as_longs, index_1.data.as_longs,
		CPU_LEN,
		SIM_DIM,
		G,
		)
