# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/baseviewer.py: Viewer base class, all viewers derive from it

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

from abc import ABC, abstractmethod
import atexit
import sys
from typing import Optional

from .debug import typechecked
from .kernel import KERNELS
from .timing import AverageTimer
from .variation import Variation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class BaseViewer(ABC):
    """
    viewer base class
    """

    def __init__(
        self,
        kernel: str,
        length: int,
        variation: Variation,
        timer_buffer: int = 20,  # average timers over this many samples
        iterations_per_frame: int = 1,
        max_iterations: Optional[int] = None,
    ):

        self._iterations_per_frame = iterations_per_frame
        self._max_iterations = max_iterations

        self._iterations = 0

        self._kernel = KERNELS[kernel]
        self._kernel.load_cls()

        self._universe = self._kernel.cls.from_galaxy(
            length=length,
            variation=variation,
        )
        self._universe.start()
        atexit.register(self._universe.stop)

        self._timer_sps = AverageTimer(timer_buffer)
        self._timer_fps = AverageTimer(timer_buffer)

        self._timer_sps.start()
        self._timer_sps.stop()

        self._timer_fps.start()

    def _exit(self):
        "terminate application"

        sys.exit()

    def _simulation(self):
        "run and measure simulation"

        self._timer_fps.stop()
        self._timer_fps.start()

        for _ in range(self._iterations_per_frame):

            self._timer_sps.start()
            self._universe.step()
            self._timer_sps.stop()

            self._iterations += 1
            if self._max_iterations is None:
                continue
            if self._iterations > self._max_iterations:
                self._exit()

    @abstractmethod
    def run(self):
        "runs viewer - must be reimplemented"
