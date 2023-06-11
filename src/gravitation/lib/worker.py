# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/worker.py: single-kernel benchmark worker

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

import gc
import traceback
from typing import Optional, Tuple

from .debug import typechecked
from .errors import WorkerError
from .kernel import KERNELS
from .logging import InfoLog, StepLog, WorkerLog
from .timing import BestRunTimer, ElapsedTimer
from .variation import Variation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Worker:
    "isolated single-kernel benchmark worker"

    def __init__(
        self,
        kernel: str,
        variation: Variation,
        length: int,
        datafile: str,
        save_after_iteration: Tuple[int, ...],
        read_initial_state: bool,
        min_iterations: int,
        min_total_runtime: int,
    ):
        WorkerLog.log(key = "start", value = WorkerLog(
            kernel = kernel,
            variation = variation,
            info = InfoLog.from_new(),
            length = length,
            status = "start",
        ).to_dict())

        self._kernel = kernel
        self._length = length
        self._datafile = datafile
        self._save_after_iteration = save_after_iteration
        self._read_initial_state = read_initial_state
        self._min_iterations = min_iterations
        if (
            len(self._save_after_iteration) > 0
            and max(self._save_after_iteration) > self._min_iterations
        ):
            self._min_iterations = max(self._save_after_iteration)
        self._min_total_runtime = min_total_runtime * 10**9  # convert to ns
        self._variation = variation

        self._iteration = 0
        self._rt = BestRunTimer()  # runtime
        self._gt = BestRunTimer()  # gc time
        self._et = None  # elapsed time, set up later

        WorkerLog.log(key = "info", value = "Creating simulation ...")

        KERNELS[self._kernel].load_cls()

        try:
            if self._read_initial_state:
                self._universe = (
                    KERNELS[self._kernel].cls.from_hdf5(
                        fn=self._datafile,
                        gn=KERNELS[self._kernel].cls.export_name_group(kernel = 'zero', length = self._length, steps = 0),
                        variation=self._variation,
                    )
                )
                assert self._length == len(self._universe)
            else:
                self._universe = (
                    KERNELS[self._kernel].cls.from_galaxy(
                        length=self._length,
                        variation=self._variation,
                    )
                )
        except Exception as e:
            self._exit(e)

        WorkerLog.log(key = "info", value = "Simulation created.")

    def _step(self):
        "run one simulation step, measure times, log values"

        try:
            self._universe.push_stage1()
            gc.collect()
            self._universe.gc_collect()
            self._rt.start()
            self._universe.step_stage1()
            rt_ = self._rt.stop()
            self._gt.start()
            gc.collect()
            gt_ = self._gt.stop()
        except Exception as e:
            self._exit(e)

        try:
            self._universe.step(stage1=False)
            self._universe.assert_not_isnan()
        except Exception as e:
            self._exit(e)

        self._iteration += 1
        if self._iteration in self._save_after_iteration:
            self._store()

        WorkerLog.log(key = "step", value = StepLog(
            iteration = self._iteration,
            runtime=rt_,
            gctime=gt_,
            runtime_min=self._rt.min,
            gctime_min=self._gt.min,
        ).to_dict())

    def _store(self):
        "store current state of simulation"

        WorkerLog.log(key = "info", value = f"Saving data after step {self._iteration:d} ...")

        try:
            self._universe.to_hdf5(
                fn=self._datafile,
                gn=self._universe.export_name_group(
                    kernel=self._kernel,
                    len=len(self._universe),
                    step=self._iteration,
                    **self._variation.to_dict(),
                )
            )
        except Exception as e:
            self._exit(e)

        WorkerLog.log(key = "info", value = f"Data saved after step {self._iteration:d}.")

    def _exit(self, error: Optional[Exception] = None):
        "exit worker process, raise WorkerError is bad result"

        if hasattr(self, "_universe"):
            self._universe.stop()

        if error is None:
            WorkerLog.log(key = "stop", value = "ok")
            return

        WorkerLog.log(key = "exit", value = traceback.format_exc())
        raise WorkerError('worker failed') from error

    def run(self):
        "run worker"

        self._universe.start()

        gc.disable()

        if 0 in self._save_after_iteration:
            self._store()

        self._et = ElapsedTimer()  # elapsed time

        # required min runs
        for _ in range(self._min_iterations):
            self._step()

        # does elapsed time satisfy min_total_runtime?
        et_ = self._et()
        if et_ >= self._min_total_runtime:
            WorkerLog.log(key = "info", value = "Minimum steps sufficient.")
            self._exit()
            return

        WorkerLog.log(key = "info", value = "Extra steps required.")
        time_remaining = self._min_total_runtime - et_
        iterations_remaining = time_remaining // et_ * self._min_iterations

        # required extra runs until min_total_runtime
        for _ in range(iterations_remaining):
            self._step()

        WorkerLog.log(key = "info", value = "Extra steps finished.")
        self._exit()
