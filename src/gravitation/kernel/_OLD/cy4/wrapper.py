# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/cy4/wrapper.py: cy4 kernel core wrapper

    Copyright (C) 2019-2023 Sebastian M. Ernst <ernst@pleiszenburg.de>

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

from array import array
import os

from .._base_ import universe_base
from .core import _step_stage1_

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class universe(universe_base):
    def start_kernel(self):
        self.DTYPE = self._dtype
        self.CDTYPE = {"float32": "f", "float64": "d"}[self.DTYPE]

        self.MASS_LEN = len(self)
        self.SIM_DIM = len(self._mass_list[0]._r)

        self.CPU_LEN = self._threads
        os.environ["OMP_NUM_THREADS"] = str(self.CPU_LEN)

        # Allocate memory: Object parameters
        self.mass_r_array = [array(self.CDTYPE) for i in range(self.SIM_DIM)]
        self.mass_v_array = [array(self.CDTYPE) for i in range(self.SIM_DIM)]
        self.mass_a_array = [array(self.CDTYPE) for i in range(self.SIM_DIM)]
        self.mass_amp_array = [array(self.CDTYPE) for i in range(self.SIM_DIM)]
        self.mass_m_array = array(self.CDTYPE)

        # Copy data from mass objects into arrays
        for pm_index, pm in enumerate(self._mass_list):
            for dim in range(self.SIM_DIM):
                self.mass_r_array[dim].append(pm._r[dim])
                self.mass_v_array[dim].append(pm._v[dim])
                self.mass_a_array[dim].append(pm._a[dim])
                for c in range(self.CPU_LEN):
                    self.mass_amp_array[dim].append(0.0)
            self.mass_m_array.append(pm._m)

        # Compute line index intervals for evenly sized batches
        total_pairs = (self.MASS_LEN * (self.MASS_LEN - 1)) // 2
        batch_length = total_pairs // self.CPU_LEN
        self.index_pool_0 = array("l")
        self.index_pool_1 = array("l")
        pair_count = 0
        start_line = 0
        for line in range(1, self.MASS_LEN - 1):
            pair_count += self.MASS_LEN - 1 - line
            if pair_count < batch_length:
                continue
            pair_count = 0
            self.index_pool_0.append(start_line)
            self.index_pool_1.append(line)
            start_line = line
        assert len(self.index_pool_0) in [(self.CPU_LEN - 1), self.CPU_LEN]
        if len(self.index_pool_0) == self.CPU_LEN - 1:
            self.index_pool_0.append(start_line)
            self.index_pool_1.append(self.MASS_LEN - 1)
        assert self.index_pool_1[-1] == self.MASS_LEN - 1

    def step_stage1(self):
        # Copy data from mass objects into arrays, reset array for a
        for pm_index, pm in enumerate(self._mass_list):
            for dim in range(self.SIM_DIM):
                self.mass_r_array[dim][pm_index] = pm._r[dim]

        # Launch cython kernel core
        _step_stage1_(
            *self.mass_r_array,
            *self.mass_a_array,
            *self.mass_amp_array,
            self.mass_m_array,
            self.index_pool_0,
            self.index_pool_1,
            self.CPU_LEN,
            self.MASS_LEN,
            self._G,
        )

        # Copy data back into mass objects into arrays
        for pm_index, pm in enumerate(self._mass_list):
            for dim in range(self.SIM_DIM):
                pm._a[dim] = self.mass_a_array[dim][pm_index]
