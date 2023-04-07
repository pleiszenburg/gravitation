# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/oc1/wrapper.py: oc1 kernel core wrapper

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

from math import sqrt
import os

import numpy as np

import oct2py

from .._base_ import universe_base

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
        self.mass_r_array = np.zeros((self.MASS_LEN, self.SIM_DIM), dtype=self.DTYPE)
        mass_m_array = np.zeros((self.MASS_LEN,), dtype=self.DTYPE)
        # Copy const data into Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            mass_m_array[pm_index] = pm._m
        # Start octave session
        self.oc = oct2py.Oct2Py(
            temp_dir="/dev/shm",
            convert_to_float=False,
        )
        self.oc.addpath(os.path.dirname(__file__))
        for name, var in [
            ("m", mass_m_array),
            ("G", self._G),
            ("SIM_DIM", self.SIM_DIM),
            ("MASS_LEN", self.MASS_LEN),
        ]:
            self.oc.eval("global %s" % name)
            self.oc.push(name, var, verbose=True)

    def step_stage1(self):
        # Copy dynamic data to Numpy infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            self.mass_r_array[pm_index, :] = pm._r[:]
        # Call octave core
        mass_a_array = self.oc.step_stage1(self.mass_r_array)
        # Push dynamic data back to Python infrastructure
        for pm_index, pm in enumerate(self._mass_list):
            pm._a[:] = mass_a_array[pm_index, :]
