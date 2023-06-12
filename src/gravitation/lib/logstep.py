# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/logstep.py: step log type

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

from typing import Any

from .debug import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class StepLog:
    "represents one single benchmark step"

    def __init__(self, iteration: int, runtime: int, gctime: int, runtime_min: int, gctime_min: int):
        self._iteration = iteration
        self._runtime = runtime
        self._gctime = gctime
        self._runtime_min = runtime_min
        self._gctime_min = gctime_min

    @property
    def iteration(self) -> int:
        "iteration of step"

        return self._iteration

    @property
    def runtime(self) -> int:
        "runtime of step"

        return self._runtime

    @property
    def gctime(self) -> int:
        "gc time of step"

        return self._iteration

    @property
    def runtime_min(self) -> int:
        "current minimal runtime"

        return self._runtime_min

    @property
    def gctime_min(self) -> int:
        "current minimal gc time"

        return self._gctime_min

    def to_dict(self) -> dict:
        "export as dict"

        return dict(
            iteration = self._iteration,
            runtime = self._runtime,
            gctime = self._gctime,
            runtime_min = self._runtime_min,
            gctime_min = self._gctime_min,
        )

    @classmethod
    def from_dict(cls, **kwargs: Any):
        "import from dict"

        return cls(**kwargs)
