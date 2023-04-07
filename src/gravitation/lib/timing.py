# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/timing.py: A few simple timers for benchmarks

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

try:
    from time import time_ns
except ImportError:  # CPython <= 3.6
    from time import time as _time

    time_ns = lambda: int(_time() * 1e9)
    del _time

from typeguard import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class TimerError(Exception):
    pass


@typechecked
class BestRunTimer:
    """looks for best runtime, infinite number of recorded runtimes / states:
    based on walltime (clocking GPU time is tricky), as little overhead as possible"""

    def __init__(self):
        self._state = []
        self._running = False
        self._lower = None

    @property
    def avg(self) -> int:
        """returns average over all recorded runtimes / states [ns as int]"""
        if len(self._state) == 0:
            raise TimerError("Nothing has been recorded.")
        return sum(self._state) // len(self._state)

    @property
    def min(self) -> int:
        """returns minimum of all recorded runtimes / states [ns as int]"""
        if len(self._state) == 0:
            raise TimerError("Nothing has been recorded.")
        return min(self._state)

    @property
    def sum(self) -> int:
        """returns sum of all recorded runtimes / states [ns as int]"""
        if len(self._state) == 0:
            raise TimerError("Nothing has been recorded.")
        return sum(self._state)

    def start(self):
        """starts a timing run"""
        if self._running:
            raise TimerError("Timer is running!")
        self._running = True
        self._lower = time_ns()

    def stop(self) -> int:
        """stops a timing run and returns runtime [ns as int]"""
        upper = time_ns()
        if not self._running:
            raise TimerError("Timer is NOT running!")
        runtime = upper - self._lower
        self._lower = None
        self._running = False
        self._state.append(runtime)
        return runtime


@typechecked
class AverageTimer(BestRunTimer):
    """looks for best runtime, limited number of recorded runtimes / states:
    based on walltime (clocking GPU time is tricky), as little overhead as possible"""

    def __init__(self, maxlen: int):
        assert maxlen >= 1
        super().__init__()
        self._maxlen = maxlen

    def stop(self) -> int:
        """stops a timing run and returns runtime [ns as int],
        ensures maximum number of runtimes"""
        runtime = super().stop()
        if len(self._state) > self._maxlen:
            self._state.pop(0)
        return runtime


@typechecked
class ElapsedTimer:
    """keeps track of time elapsed since initialization"""

    def __init__(self):
        self._state = time_ns()

    def __call__(self) -> int:
        return time_ns() - self._state
