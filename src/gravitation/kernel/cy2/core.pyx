# -*- coding: utf-8 -*-
# cython: language_level=3, boundscheck=False

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/cy2/core.pyx: cy2 kernel cython core

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
	float *rx, float *ry, float *rz,
	float *ax, float *ay, float *az,
	float *m,
	long SIM_DIM,
	float G,
	):

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

	ax[index_i] -= relative_rx * a1
	ay[index_i] -= relative_ry * a1
	az[index_i] -= relative_rz * a1

	ax[index_j] += relative_rx * a2
	ay[index_j] += relative_ry * a2
	az[index_j] += relative_rz * a2

cdef void _step_stage1_c_(
	float *rx, float *ry, float *rz,
	float *ax, float *ay, float *az,
	float *m,
	long SIM_DIM,
	float G,
	):

	# iteration index variables
	cdef long index_i
	cdef long index_j

	# reset a to zero
	memset(ax, 0, SIM_DIM * sizeof(float))
	memset(ay, 0, SIM_DIM * sizeof(float))
	memset(az, 0, SIM_DIM * sizeof(float))

	# update all unique pairs
	index_i = 0
	while index_i < (SIM_DIM - 1):
		index_j = index_i + 1
		while index_j < SIM_DIM:
			_update_pair_c_(
				index_i, index_j,
				rx, ry, rz,
				ax, ay, az,
				m,
				SIM_DIM,
				G,
				)
			index_j += 1
		index_i += 1

def _step_stage1_(
	array.array rx, array.array ry, array.array rz,
	array.array ax, array.array ay, array.array az,
	array.array m,
	long SIM_DIM,
	float G,
	):

	_step_stage1_c_(
		rx.data.as_floats, ry.data.as_floats, rz.data.as_floats,
		ax.data.as_floats, ay.data.as_floats, az.data.as_floats,
		m.data.as_floats,
		SIM_DIM,
		G,
		)
