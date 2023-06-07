# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/const.py: Const values

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
# IMPORTS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from enum import (
    Enum as _Enum,
    auto as _auto,
)
from psutil import cpu_count as _cpu_count

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONSTS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Dtype(_Enum):
    float32 = _auto()
    float64 = _auto()

class State(_Enum):
    preinit = _auto()
    started = _auto()
    stopped = _auto()

class Stream(_Enum):
    stdout = _auto()
    stderr = _auto()

class Target(_Enum):
    cpu = _auto()
    gpu = _auto()

Threads = _Enum('Threads', [
    ('auto', 0),  # auto
    ('single', 1),
    ('physical', _cpu_count(logical = False)),
    ('logical', _cpu_count(logical = True)),
    *[
        (f't{_idx:d}', _idx)
        for _idx in range(1, _cpu_count(logical = True) + 1)
    ],
])
Threads.modes = lambda: (
    _item
    for _item in Threads
    if not _item.name.startswith('_') and _item is not Threads.auto
)

DEFAULT_DTYPE = Dtype.float64
DEFAULT_TARGET = Target.cpu
DEFAULT_THREADS = Threads.single

DEFAULT_LEN = 2048

DIMS = 3
